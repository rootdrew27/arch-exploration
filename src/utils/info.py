
def view_modules(model):
    for i, param_tensor in enumerate(model.state_dict()):
        print(i, ' ' * (3 - len(str(i)) ), param_tensor, " " * (70 - len(param_tensor)), model.state_dict()[param_tensor].size())