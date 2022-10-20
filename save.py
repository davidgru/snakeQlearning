import torch


def save(model, path):
    model = torch.jit.script(model)
    model.save(path)