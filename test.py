import SparseGenerater as sparse
import Algorithms as alg
import torch
import numpy as np
import matplotlib.pyplot as plt

x_length = 8000
y_length = 1600
batch_size = 1
iteration = 250
sparsitys = [0.85, 0.9, 0.95, 0.98]
lines = []
device = 'cuda'
for sparsity in sparsitys:
    x_origin = sparse.generate_sparse_signal([batch_size, x_length, 1], sparsity, type='int', device=device)
    mat = sparse.generate_normalized_uniform_matrix([batch_size, y_length, x_length], device=device)
    y_detect = mat.matmul(x_origin)
    x_recover, MSE_amp = alg.amp(y_detect, mat, x_origin, theta=0.1, iteration=iteration, device=device)
    # x_recover, MSE_fista = alg.fista(y_detect, mat, x_origin, alpha=0.01, iteration=iteration, device=device)
    # x_recover, MSE_ista = alg.ista(y_detect, mat, x_origin, alpha=0.01, iteration=iteration, device=device)
    line, = plt.plot(list(range(len(MSE_amp))), MSE_amp)
    lines = np.append(lines, line)
# x_recover, MSE_fista = alg.fista(y_detect, mat, x_origin, alpha=0.01, Lambda=1, iteration=iteration, device=device)
# x_recover, MSE_ista = alg.ista(y_detect, mat, x_origin, alpha=0.01, Lambda=1, iteration=iteration, device=device)

# l1, = plt.plot(list(range(len(MSE_fista))), MSE_fista)
# l2, = plt.plot(list(range(len(MSE_ista))), MSE_ista)
# l3, = plt.plot(list(range(len(MSE_amp))), MSE_amp)
# plt.legend(handles=lines, labels=['0.6', '0.7', '0.8', '0.9'], loc='best')
plt.ylabel('MSE')
plt.xlabel('iteration')
plt.title('MSE over iterations')
plt.show()
