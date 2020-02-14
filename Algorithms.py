import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def proximal(y, tau, x0=0, Lambda=1, tau0=1, phi='l1'):
    y = y.squeeze(2)
    x = torch.zeros_like(y)
    if phi == 'l1':
        t = tau * Lambda
        x[y > t] = (y - t)[y > t]
        x[y < -t] = (y + t)[y < -t]
    elif phi == 'l2':
        if x0 == 0:
            x0 = torch.zeros_like(y)
        x = (tau * x0 + tau0 * y) / (tau + tau0)
    return x.unsqueeze(2)


def soft_threshold_gradient(y, tau, Lambda=1):
    grad = torch.zeros_like(y)
    t = tau * Lambda
    grad[y > t] = 1
    grad[y < -t] = 1
    return grad.sum(dim=1)


def ista(y, A, true_x, alpha, Lambda=1, iteration=100, device='cpu'):
    try:
        M = A.size()[2]
        batch_size = A.size()[0]
    except IndexError:
        M = A.size()[1]
        batch_size = 1
    x = torch.zeros(batch_size, M, 1, device=device)
    MSE = torch.zeros(iteration, device=device)
    mseloss = nn.MSELoss(reduction='mean')
    stop_iter = iteration
    for i in tqdm(range(iteration)):
        MSE_tmp = mseloss(x, true_x)
        if i > 1 and MSE_tmp >= MSE[i - 1]:
            stop_iter = i
            break
        MSE[i] = MSE_tmp
        d = y - A.matmul(x)
        r = x + alpha * A.permute(0, 2, 1).matmul(d)
        x = proximal(r, alpha, phi='l1', Lambda=Lambda)
    if device == 'cuda':
        MSE = MSE.cpu()
    return x, MSE[0:stop_iter]


def fista(y, A, true_x, alpha, Lambda=1, iteration=100, device='cpu'):
    try:
        M = A.size()[2]
        batch_size = A.size()[0]
    except IndexError:
        M = A.size()[1]
        batch_size = 1
    x = torch.zeros(batch_size, M, 1, device=device)
    epsilon = torch.zeros(batch_size, 1, device=device)
    MSE = torch.zeros(iteration, device=device)
    mseloss = nn.MSELoss(reduction='mean')
    stop_iter = iteration
    for i in tqdm(range(iteration)):
        MSE_tmp = mseloss(x, true_x)
        if i > 1 and MSE_tmp >= MSE[i - 1]:
            stop_iter = i
            break
        MSE[i] = MSE_tmp
        epsilon1 = ((4 * (epsilon ** 2) + 1).sqrt() + 1) / 2
        gamma = (epsilon - 1) / epsilon1
        epsilon = epsilon1
        d = y - A.matmul(x)
        r = x + alpha * A.permute(0, 2, 1).matmul(d)
        x1 = proximal(r, alpha, phi='l1', Lambda=Lambda)
        x = x1 + (gamma * (x1 - x).squeeze()).unsqueeze(2)
    if device == 'cuda':
        MSE = MSE.cpu()
    return x, MSE[0:stop_iter]


def amp(y, A, true_x, theta, Lambda=1, iteration=100, device='cpu'):
    try:
        M = A.size()[2]
        N = A.size()[1]
        batch_size = A.size()[0]
    except IndexError:
        M = A.size()[1]
        N = A.size()[0]
        batch_size = 1
    x = torch.zeros(batch_size, M, 1, device=device)
    alpha = 0
    d = torch.zeros_like(y)
    MSE = torch.zeros(iteration, device=device)
    mseloss = nn.MSELoss(reduction='mean')
    stop_iter = iteration
    for i in tqdm(range(iteration)):
        MSE_tmp = mseloss(x, true_x)
        if i > 1 and MSE_tmp >= MSE[i - 1]:
           stop_iter = i
           break
        MSE[i] = MSE_tmp
        try:
            d = d.squeeze(2)
        except IndexError:
            d = d
        d = y - A.matmul(x) + ((M / N) * alpha * d).unsqueeze(2)
        r = x + theta * A.permute(0, 2, 1).matmul(d)
        t = theta * d.norm(dim=1) / np.sqrt(N)
        x = proximal(r, t, phi='l1', Lambda=1)
        alpha = soft_threshold_gradient(r, theta) / M
    if device == 'cuda':
        MSE = MSE.cpu()
    return x, MSE[0:stop_iter]
