import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from SparseGenerater import generate_sparse_signal
import matplotlib.pyplot as plt


class AutoEncoder(nn.Module):
    def __init__(self, N, L, Q):
        super(AutoEncoder, self).__init__()
        self.real_encode = nn.Linear(N, L)
        self.imag_encode = nn.Linear(N, L)
        self.decode = nn.Sequential(
            nn.Linear(2 * L, Q),
            nn.ReLU(inplace=True),
            nn.Linear(Q, Q),
            nn.ReLU(inplace=True),
            nn.Linear(Q, N),
            nn.Sigmoid()
        )
        self.N = N
        self.L = L

    def forward(self, real, imag):
        real_real = self.real_encode(real)
        real_imag = self.real_encode(imag)
        imag_real = self.imag_encode(real)
        imag_imag = self.imag_encode(imag)
        real_y = real_real - imag_imag
        imag_y = real_imag + imag_real
        y = torch.cat((real_y, imag_y), dim=1)
        y = self.decode(y)
        return y


accs = np.zeros([50, 50])
for i, ratio in enumerate(np.arange(1, 0, -0.02)):
    batch_size = 1000
    N = 100
    L = int(N * ratio)
    Q = 40
    EPOCH = 100
    LR = 0.001
    for j, sparsity in enumerate(np.arange(1, 0, -0.02)):
        print(sparsity)
        data = generate_sparse_signal([batch_size, N], sparsity, type='complex', device='cuda')
        data_real = data[:, :N]
        data_imag = data[:, N:]
        alpha = torch.zeros(batch_size, N, device='cuda')
        alpha[data_real > 0] = 1
        net = AutoEncoder(N, L, Q).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        loss_func = nn.BCELoss()

        for epoch in tqdm(range(EPOCH)):
            recovery = net(data_real, data_imag)
            loss = loss_func(recovery, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        max_accuracy = 0
        optim_threshold = 0
        for threshold in np.arange(0, 1, 0.01):
            predict = recovery.detach().clone()
            predict[predict < threshold] = 0
            predict[predict >= threshold] = 1
            accuracy = (predict.long() == alpha.long()).sum().item() / (batch_size * N)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                optim_threshold = threshold
        print('optimal threshold: ', optim_threshold)
        print('max accuracy: ', max_accuracy)
        accs[i, j] = max_accuracy

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
x = np.arange(1, 0, -0.02)
y = np.arange(1, 0, -0.02)
x, y = np.meshgrid(x, y)

# Plot the surface.
surf = ax.plot_surface(x, y, np.log(accs), cmap=plt.cm.coolwarm, linewidth=0.1, antialiased=False)
plt.show()

