import SparseGenerater as sparse
import Algorithms as alg
import torch
import matplotlib.pyplot as plt

x_length = 8000
y_length = 1600
batch_size = 5
iteration = 400
x_origin = sparse.generate_sparse_signal([batch_size, x_length, 1], 0.9, upperbound=10)
mat = sparse.generate_normalized_gaussian_matrix([batch_size, y_length, x_length])
y_detect = mat.matmul(x_origin)
x_recover, MSE_amp = alg.amp(y_detect, mat, x_origin, theta=0.1, Lambda=1, iteration=iteration)
x_recover, MSE_fista = alg.fista(y_detect, mat, x_origin, alpha=0.01, Lambda=1, iteration=iteration)
x_recover, MSE_ista = alg.ista(y_detect, mat, x_origin, alpha=0.01, Lambda=1, iteration=iteration)

l1, = plt.plot(list(range(len(MSE_fista))), MSE_fista)
l2, = plt.plot(list(range(len(MSE_ista))), MSE_ista)
l3, = plt.plot(list(range(len(MSE_amp))), MSE_amp)
plt.legend(handles=[l1, l2, l3], labels=['MSE_FISTA', 'MSE_ISTA', 'MSE_AMP'], loc='best')
plt.ylabel('MSE')
plt.xlabel('iteration')
plt.title('MSE over iterations')
plt.show()
