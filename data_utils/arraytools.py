"""
Functions to convert from one type to another type
"""
import torch as t
import numpy as np
from torchvision.transforms import functional as F


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.cpu().numpy().transpose(1,2,0)
    if isinstance(data, t.autograd.Variable):
        return tonumpy(data.data)


def totensor(data, cuda=False):
    if isinstance(data, np.ndarray):
        tensor = F.to_tensor(data)
    if isinstance(data, t.Tensor):
        tensor = data
    if isinstance(data, t.autograd.Variable):
        tensor = data.data
    if cuda:
        tensor = tensor.cuda()
    return tensor


def tovariable(data):
    if isinstance(data, np.ndarray):
        return tovariable(totensor(data))
    if isinstance(data, t.Tensor):
        return t.autograd.Variable(data)
    if isinstance(data, t.autograd.Variable):
        return data
    else:
        raise ValueError("UnKnow data type: %s, input should be {np.ndarray,Tensor,Variable}" %type(data))


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        return data.view(1)[0]
    if isinstance(data, t.autograd.Variable):
        return data.data.view(1)[0]