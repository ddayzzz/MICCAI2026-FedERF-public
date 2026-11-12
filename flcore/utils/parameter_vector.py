import torch

def get_mdl_params(model, device="cpu"):
    vec = []
    for param in model.parameters():
        vec.append(param.clone().detach().to(device).reshape(-1))
    return torch.cat(vec)

def set_model_from_params(model, params):
    idx = 0
    for param in model.parameters():
        length = param.numel()
        param.data.copy_(params[idx:idx + length].reshape(param.shape))
        idx += length
    return model
