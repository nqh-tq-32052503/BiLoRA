
import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.models.helpers import named_apply, checkpoint_seq
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_

from models.vit_base import LayerScale, get_init_weights_vit, _load_weights, init_weights_vit_timm, Block

_logger = logging.getLogger(__name__)


class QuantizeAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_embeddings=50, alpha=0.05):
        super().__init__()
        self.num_heads = num_heads
        self.num_embeddings = num_embeddings
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.init_embedding_spaces()

    def init_embedding_spaces(self):
        self.q_embed_space = nn.Parameter(torch.randn(self.num_heads, self.num_embeddings, self.head_dim), requires_grad=True)
        self.k_embed_space = nn.Parameter(torch.randn(self.num_heads, self.num_embeddings, self.head_dim), requires_grad=True)

    def quantize(self, unquantized_tensor: torch.Tensor, embedding_space: torch.Tensor):
        batch_size = unquantized_tensor.size(0)
        dist_matrix = torch.cdist(unquantized_tensor, embedding_space.unsqueeze(0), p=2) # Shape: (7, 12, 197, 50)
        selected_indices = torch.argmin(dist_matrix, dim=-1)
        B_expanded = embedding_space.unsqueeze(0).expand(batch_size, -1, -1, -1)
        indices_expanded = selected_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        quantized_tensor = torch.gather(B_expanded, 2, indices_expanded)
        return quantized_tensor

    def quantize_except_cls_token(self, unquantized_tensor: torch.Tensor, embedding_space: torch.Tensor):
        batch_size = unquantized_tensor.size(0)
        cls_token = unquantized_tensor[:, :, 0, :].unsqueeze(2)
        feature_token = unquantized_tensor[:, :, 1:, :]
        dist_matrix = torch.cdist(feature_token, embedding_space.unsqueeze(0), p=2) # Shape: (7, 12, 197, 50)
        selected_indices = torch.argmin(dist_matrix, dim=-1)
        B_expanded = embedding_space.unsqueeze(0).expand(batch_size, -1, -1, -1)
        indices_expanded = selected_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        quantized_feature = torch.gather(B_expanded, 2, indices_expanded)
        return cls_token, quantized_feature

    def forward(self, x, task_id=None, register_hook=False, prompt=None, get_feat=False,get_cur_feat=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat((pk,k), dim=2)
            v = torch.cat((pv,v), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Quantize
        q_cls, quant_q = self.quantize_except_cls_token(q, self.q_embed_space)
        k_cls, quant_k = self.quantize_except_cls_token(k, self.k_embed_space)
        cat_q = torch.cat((q_cls.detach(), quant_q), dim=2)
        cat_k = torch.cat((k_cls.detach(), quant_k), dim=2)
        quant_attn = (cat_q @ cat_k.transpose(-2, -1)) * self.scale
        quant_attn = quant_attn.softmax(dim=-1)

        # Quantize Loss
        quant_loss = F.mse_loss(quant_q, q[:, :, 1:, :].detach()) + F.mse_loss(quant_k, k[:, :, 1:, :].detach())
        quant_loss += F.kl_div(quant_attn, attn.detach())


        quant_attn = self.attn_drop(quant_attn)
                
        if register_hook:
            self.save_attention_map(quant_attn)
            quant_attn.register_hook(self.save_attn_gradients)        

        x = (quant_attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        self.locals = locals()
        return x, quant_loss



class CPEViT(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init: (str): weight init scheme
            init_values: (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,n_tasks=n_tasks,r=rank)
            for i in range(depth)])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Representation layer. Used for original ViT models w/ in21k pretraining.
        self.representation_size = representation_size
        self.pre_logits = nn.Identity()
        if representation_size:
            self._reset_representation(representation_size)

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()
        self.out_dim = final_chs

        if weight_init != 'skip':
            self.init_weights(weight_init)
        # CPE
        self.cpe = nn.Parameter(torch.randn(10, self.embed_dim), requires_grad=True)

    def _reset_representation(self, representation_size):
        self.representation_size = representation_size
        if self.representation_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, self.representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None, representation_size=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        if representation_size is not None:
            self._reset_representation(representation_size)
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, task_id=None, labels=None):
        y1 = self.patch_embed(x)
        if labels is not None:
            selected_cpe = torch.index_select(self.cpe, 0, labels)
            y2 = y1 + selected_cpe.unsqueeze(1)
        y3 = torch.cat((self.cls_token.expand(y2.shape[0], -1, -1), y2), dim=1)

        y4 = self.pos_drop(y3 + self.pos_embed)
        for blk in self.blocks:
            y4 = blk(y4, task_id=task_id, register_hook=False)
        y5 = self.norm(y4)
        self.locals = locals()
        return y5


    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.pre_logits(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, task_id=None, labels=None, register_blk=None, get_feat=False, get_cur_feat=False):
        x = self.forward_features(x, task_id=task_id, labels=labels)
        return x, None

class QuantViT(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64, num_embeddings=256):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init: (str): weight init scheme
            init_values: (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,n_tasks=n_tasks,r=rank)
            for i in range(depth)])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Representation layer. Used for original ViT models w/ in21k pretraining.
        self.representation_size = representation_size
        self.pre_logits = nn.Identity()
        if representation_size:
            self._reset_representation(representation_size)

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()
        self.out_dim = final_chs

        if weight_init != 'skip':
            self.init_weights(weight_init)
        # CPE
        self.num_embeddings = num_embeddings
        self.init_cpe()
        self.quant_embedding = nn.Parameter(torch.randn(num_embeddings, self.embed_dim), requires_grad=True)

    def _reset_representation(self, representation_size):
        self.representation_size = representation_size
        if self.representation_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, self.representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    def init_cpe(self, checkpoint="cpe.pt"):
        weight = torch.load(checkpoint, weights_only=False, map_location="cpu")
        self.cpe = weight
        self.cpe.requires_grad = False

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None, representation_size=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        if representation_size is not None:
            self._reset_representation(representation_size)
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, task_id=None, labels=None):
        x = self.patch_embed(x)
        if labels is not None:
            selected_cpe = torch.index_select(self.cpe.weight, 0, labels)
            x = x + selected_cpe.unsqueeze(1)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        for blk in self.blocks:
            x = blk(x, task_id=task_id, register_hook=False)

        return x


    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.pre_logits(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, task_id=None, labels=None, register_blk=None, get_feat=False, get_cur_feat=False):
        x = self.forward_features(x, task_id=task_id, labels=labels)
        return x, None