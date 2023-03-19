import torch
import torch.nn.functional as F
import os
import numpy as np
from torch.utils import data
from functools import reduce
import operator

def init_hidden(batch_size, cell_size):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell


def CUDA_or_not(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()
    return obj


def gumbel_softmax(training, x, tau=1.0, hard=False):
    if training:
        eps = 1e-20
        U = torch.rand(x.size()).cuda()
        U = -torch.log(-torch.log(U + eps) + eps)
        r_t = x + 0.5*U
        r_t = F.softmax(r_t / tau, dim=-1)

        if not hard:
            return r_t
        else:
            shape = r_t.size()
            _, ind = r_t.max(dim=-1)
            r_hard = torch.zeros_like(r_t).view(-1, shape[-1])
            r_hard.scatter_(1, ind.view(-1, 1), 1)
            r_hard = r_hard.view(*shape)
            return (r_hard - r_t).detach() + r_t
    else:
        selected = torch.zeros_like(x)
        Q_t = torch.argmax(x, dim=1).unsqueeze(1)
        selected = selected.scatter(1, Q_t, 1)
        return selected.float()





def find_length(list_tensors):
    """find the length of list of tensors"""
    if type(list_tensors[0]) is np.ndarray:
        length = [x.shape[0] for x in list_tensors]
        fea_dim = [x.shape[1] for x in list_tensors if x.size(0) > 0]
    else:
        length = [x.size(0) for x in list_tensors]
        fea_dim = [x.shape[1] for x in list_tensors if x.size(0) > 0]
    return length, fea_dim


def pad_tensor(tensor, length):
    """Pad a tensor, given by the max length"""
    if tensor.size(0) == length:
        return tensor
    return torch.cat([tensor, tensor.new(length - tensor.size(0),
                                  *tensor.size()[1:]).zero_()])


def pad_list_tensors(list_tensor, max_length=None):
    """Pad a list of tensors and return a list of tensors"""
    tensor_length, fea_dim = find_length(list_tensor)

    fea_dim = max(fea_dim)
    if max_length is None:
        max_length = max(tensor_length)

    list_padded_tensor = []
    for tensor in list_tensor:
        if tensor.size(0) == 0:
            tensor = torch.zeros(1, fea_dim)
        if tensor.size(0) != max_length:
            tensor = pad_tensor(tensor, max_length)
        list_padded_tensor.append(tensor)
    return torch.stack(list_padded_tensor), tensor_length


def create_mask(batchsize, max_length, length):
    """Given the length create a mask given a padded tensor"""
    tensor_mask = torch.zeros(batchsize, max_length)
    for idx, row in enumerate(tensor_mask):
        row[:length[idx]] = 1
    return tensor_mask

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

