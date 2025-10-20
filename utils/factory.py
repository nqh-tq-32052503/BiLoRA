from methods.bilora import BiLoRA

def get_model(model_name, args):
    name = model_name.lower()
    options = {
               'bilora': BiLoRA,
               }
    return options[name](args)

