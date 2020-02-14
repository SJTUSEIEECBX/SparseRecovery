import torch
import numpy as np


def generate_sparse_signal(size, sparsity, type='float', lowerbound=0, upperbound=1, device='cpu'):
    prob = torch.rand(size, device=device)
    prob[prob < sparsity] = 0
    prob[prob > sparsity] = 1
    if type == 'float':
        val = (torch.rand(size, device=device).mul(upperbound - lowerbound) + lowerbound).mul(prob)
    elif type == 'bool':
        val = prob.long()
    elif type == 'complex':
        val_real = (torch.rand(size, device=device).mul(upperbound - lowerbound) + lowerbound).mul(prob)
        val_imag = (torch.rand(size, device=device).mul(upperbound - lowerbound) + lowerbound).mul(prob)
        val = torch.cat((val_real, val_imag), dim=1)
    else:
        val = prob.long().mul(torch.randint(0, 2, size, device=device) * 2 - 1).to(dtype=torch.float32)
    return val


def generate_normalized_gaussian_matrix(size, device='cpu'):
    mat = torch.randn(size, device=device)
    try:
        N = size[2]
        M = size[1]
    except IndexError:
        M = size[0]
    mat = mat / np.sqrt(M)
    return mat


def generate_normalized_uniform_matrix(size, device='cpu'):
    mat = torch.rand(size, device=device) * 2 - 1
    try:
        N = size[2]
        M = size[1]
    except IndexError:
        M = size[0]
    mat = mat / np.sqrt(M)
    return mat
