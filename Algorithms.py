import torch
import numpy as np
from tqdm import tqdm


def proximal(y, tau, x0=0, Lambda=1, tau0=1, phi='l1'):
    x = torch.zeros_like(y)
    if phi == 'l1':
        t = tau * Lambda
        x[y > t] = y[y > t] - t
        x[y < -t] = y[y < -t] + t
    elif phi == 'l2':
        if x0 == 0:
            x0 = torch.zeros_like(y)
        x = (tau * x0 + tau0 * y) / (tau + tau0)
    return x


def soft_threshold_gradient(y, tau, Lambda=1):
    grad = torch.zeros_like(y)
    t = tau * Lambda
    grad[y > t] = 1
    grad[y < -t] = 1
    return grad.sum(dim=1)


def ista(y, A, true_x, alpha, Lambda=1, iteration=100):
    try:
        M = A.size()[2]
        batch_size = A.size()[0]
    except IndexError:
        M = A.size()[1]
        batch_size = 1
    x = torch.zeros(batch_size, M, 1)
    MSE = torch.zeros(iteration)
    for i in tqdm(range(iteration)):
        d = y - A.matmul(x)
        r = x + alpha * A.permute(0, 2, 1).matmul(d)
        x = proximal(r, alpha, phi='l1', Lambda=Lambda)
        MSE[i] = (x - true_x).norm() / (batch_size * M)
    return x, MSE


def fista(y, A, true_x, alpha, Lambda=1, iteration=100):
    try:
        M = A.size()[2]
        batch_size = A.size()[0]
    except IndexError:
        M = A.size()[1]
        batch_size = 1
    x = torch.zeros(batch_size, M, 1)
    epsilon = torch.zeros(batch_size, 1)
    MSE = torch.zeros(iteration)
    for i in tqdm(range(iteration)):
        epsilon1 = ((4 * (epsilon ** 2) + 1).sqrt() + 1) / 2
        gamma = (epsilon - 1) / epsilon1
        epsilon = epsilon1
        d = y - A.matmul(x)
        r = x + alpha * A.permute(0, 2, 1).matmul(d)
        x1 = proximal(r, alpha, phi='l1', Lambda=Lambda)
        x = x1 + (gamma * (x1 - x).squeeze()).unsqueeze(2)
        MSE[i] = (x - true_x).norm() / (batch_size * M)
    return x, MSE.numpy()


def amp(y, A, true_x, theta, Lambda=1, iteration=100):
    try:
        M = A.size()[2]
        N = A.size()[1]
        batch_size = A.size()[0]
    except IndexError:
        M = A.size()[1]
        N = A.size()[0]
        batch_size = 1
    x = torch.zeros(batch_size, M, 1)
    alpha = 0
    d = torch.zeros_like(y)
    MSE = torch.zeros(iteration)
    for i in tqdm(range(iteration)):
        d = y - A.matmul(x) + ((N / M) * alpha * d.squeeze()).unsqueeze(2)
        r = x + theta * A.permute(0, 2, 1).matmul(d)
        x = proximal(r, theta, phi='l1', Lambda=Lambda)
        alpha = soft_threshold_gradient(r, theta) / N
        MSE[i] = (x - true_x).norm() / (batch_size * M)
    return x, MSE
