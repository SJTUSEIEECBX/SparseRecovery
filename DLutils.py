import torch
import torch.nn as nn
import numpy as np

def cpxmm(a_real, a_imag, b_real, b_imag):
    res_real = a_real.matmul(b_real) - a_imag.matmul(b_imag)
    res_imag = a_real.matmul(b_imag) + a_imag.matmul(b_real)
    return res_real, res_imag


class ComplexLinear(nn.Module):
    def __init__(self, M, N, activation=None):
        super(ComplexLinear, self).__init__()
        self.real = nn.Linear(M // 2, N // 2)
        self.imag = nn.Linear(M // 2, N // 2)
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

    def forward(self, x):
        x_real = x[:, :x.shape[1] // 2]
        x_imag = x[:, x.shape[1] // 2:]
        real_real = self.real(x_real)
        real_imag = self.real(x_imag)
        imag_real = self.imag(x_real)
        imag_imag = self.imag(x_imag)
        real_y = real_real - imag_imag
        imag_y = real_imag + imag_real
        y = torch.cat((real_y, imag_y), dim=1)
        if self.activation != None:
            y = self.activation(y)
        return y


def complexmatmul(X, Y):
    X_real = X[:, :X.shape[1] // 2, :]
    X_imag = X[:, X.shape[1] // 2:, :]
    Y_real = Y[:, :Y.shape[1] // 2, :]
    Y_imag = Y[:, Y.shape[1] // 2:, :]
    res_real = Y_real.matmul(X_real) - Y_imag.matmul(X_imag)
    res_imag = Y_real.matmul(X_imag) + Y_imag.matmul(X_real)
    res = torch.cat((res_real, res_imag), dim=1)
    return res


class SimpleResNet(nn.Module):
    def __init__(self, N_bs, T, K):
        super(SimpleResNet, self).__init__()
        self.inputsize = 2 * N_bs * T
        self.outputsize = 2 * N_bs * K
        self.N_bs = N_bs
        self.T = T
        self.K = K
        self.block1 = ComplexLinear(self.inputsize, self.inputsize)
        self.block2 = ComplexLinear(self.inputsize, self.inputsize)
        self.block3 = nn.Sequential(
            ComplexLinear(self.outputsize, 1000),
            ComplexLinear(1000, self.outputsize),
        )
        # self.block4 = ComplexLinear(self.outputsize, self.outputsize)

    def forward(self, Y, P):
        Y = Y.view(Y.shape[0], -1)
        Y1 = self.block1(Y)
        Y1 = Y + Y1
        Y2 = self.block2(Y1)
        Y2 = Y1 + Y2
        Y2 = Y2.reshape(Y2.shape[0], 2 * self.T, -1)
        X = complexmatmul(Y2, P)
        X = X.view(X.shape[0], -1)
        X1 = self.block3(X)
        X = X + X1
        # X2 = self.block4(X1)
        # X2 = X2 + X1
        # X2 = X2.reshape(X2.shape[0], 2 * self.N_bs, -1)
        X = X.reshape(X.shape[0], 2 * self.K, -1)
        return X


def totensor(x, cuda=False):
    x_real = torch.from_numpy(np.real(x)).squeeze().float()
    x_imag = torch.from_numpy(np.imag(x)).squeeze().float()
    if cuda:
        X_train = torch.cat((x_real, x_imag), dim=1).cuda()
    else:
        X_train = torch.cat((x_real, x_imag), dim=1)
    return X_train


def toarray(x, cuda=False):
    x = x.detach()
    if cuda:
        x = x.cpu()
    x = x.numpy()
    x = x[:, :x.shape[1] // 2] + x[:, x.shape[1] // 2:] * 1j
    return x


class UnbalanceBCELoss(nn.Module):
    def __init__(self, p=1):
        super(UnbalanceBCELoss, self).__init__()
        self.p = p
    
    def forward(self, predict, target):
        pn = self.p / (self.p + 1)
        loss = -pn * target * torch.log(predict + 1e-5) - (1 - pn) * (1 - target) * torch.log(1 - predict + 1e-5)
        loss = loss.mean()
        return loss
    
    def dec_p(self, rate=1):
        if self.p > rate:
            self.p -= rate
        else:
            print('cannot decrease p any more')
            
    def inc_p(self, rate):
        self.p += rate
