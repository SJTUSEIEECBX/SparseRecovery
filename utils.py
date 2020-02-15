import numpy as np
from tqdm import tqdm_notebook
import scipy.special as sp


def generate_channel_single_user(N_bs, N_ms, fc, Lp, sigma2alpha, fs, K, P_set):
    n_bs = np.expand_dims(np.arange(N_bs), axis=1)
    idx_bs = np.expand_dims(np.arange(Lp), axis=0) + np.random.randint(N_bs) + 1
    idx_bs[idx_bs > N_bs] -= N_bs
    theta_bs = idx_bs / N_bs
    A_ms = np.ones((1, Lp))
    A_bs = np.exp(-2j * np.pi * n_bs.dot(theta_bs))
    L_cp = K / 4
    tau_max = L_cp / fs
    tau = np.sort(tau_max * np.random.rand(1, Lp)) * (-2 * np.pi * fs / K)
    alpha = np.sort(np.sqrt(sigma2alpha / 2) * (np.random.randn(1, Lp) + 1j * np.random.randn(1, Lp)))
    A_r = np.fft.fft(np.eye(N_bs)) / np.sqrt(N_bs)
    A_t = np.fft.fft(np.eye(N_ms)) / np.sqrt(N_ms)
    H_frq = np.zeros((N_bs, N_ms, len(P_set)), dtype=complex)
    H_ang = np.zeros((N_bs, N_ms, len(P_set)), dtype=complex)
    for i in range(len(P_set)):
        D = np.diag((alpha * np.exp(1j * P_set[i] * tau)).squeeze())
        H_frq[:, :, i] = A_bs.dot(D).dot(A_ms.T)
        H_ang[:, :, i] = A_r.T.dot(H_frq[:, :, i]).dot(A_t)
    return H_frq, H_ang, theta_bs


def generate_active_user(K, Ka):
    activity = np.zeros((K, 1))
    idx = np.random.permutation(K)
    activity[idx[:Ka]] = 1
    return activity


def generate_active_channel(K, Ka, N_bs, P_set, Lp_min, Lp_max, N_ms, fc, sigma2alpha, fs):
    P = len(P_set)
    H_frq = np.zeros((K, N_bs, P), dtype=complex)
    H_ang = np.zeros((K, N_bs, P), dtype=complex)
    activity = generate_active_user(K, Ka)
    for user in range(K):
        Lp = np.random.randint(Lp_min, Lp_max)
        H_f, H_a, _ = generate_channel_single_user(N_bs, N_ms, fc, Lp, sigma2alpha, fs, K, P_set)
        H_frq[user] = activity[user] * np.transpose(H_f, (1, 0, 2))
        H_ang[user] = activity[user] * np.transpose(H_a, (1, 0, 2))
    return H_frq, H_ang, activity


def generate_batched_active_channel(params):
    H_f = np.zeros((params['simulations'], params['K'], params['N_bs'], params['P']), dtype=complex)
    H_a = np.zeros((params['simulations'], params['K'], params['N_bs'], params['P']), dtype=complex)
    act = np.zeros((params['simulations'], params['K']))
    for i in range(params['simulations']):
        H_frq, H_ang, activity = generate_active_channel(params['K'],
                                                         params['Ka'],
                                                         params['N_bs'],
                                                         params['P_set'],
                                                         params['Lp_min'],
                                                         params['Lp_max'],
                                                         params['N_ms'],
                                                         params['fc'],
                                                         params['sigma2alpha'],
                                                         params['fs'])
        H_f[i] = H_frq
        H_a[i] = H_ang
        act[i] = activity.squeeze()
    return H_f, H_a, act


def generate_random_pilot(T, K, P, batch):
    pilot = (np.random.randn(batch, T, K, P) + 1j * np.random.randn(batch, T, K, P)) / np.sqrt(2)
    return pilot


def pilot_through_channel(pilot, H_f, nvar):
    batch_size, K, N_bs, P = H_f.shape
    _, T, _, _ = pilot.shape
    y = np.zeros((batch_size, T, N_bs, P), dtype=complex)
    for sim in range(batch_size):
        for p in range(P):
            noise = np.sqrt(nvar / 2) * (np.random.randn(T, N_bs) + 1j * np.random.randn(T, N_bs))
            y[sim, :, :, p] = pilot[sim, :, :, p].dot(H_f[sim, :, :, p]) + noise
    return y


def dmmv_amp_detection(y, pilot):
    iters = 200
    tol = 1e-5
    snr0 = 100
    damp = 0.3
    batch_size, G, M, P = y.shape
    _, _, K, _ = pilot.shape
    d = G / K
    normal_cdf = lambda x: 0.5 * (1 + sp.erf(x / np.sqrt(2)))
    normal_pdf = lambda x: np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    alpha_grid = np.linspace(0, 10, 1024)
    rho_se = (1 - (2 / d) * ((1 + alpha_grid ** 2) * normal_cdf(-alpha_grid) - alpha_grid * normal_pdf(alpha_grid))) / \
             (1 + alpha_grid ** 2 - 2 * (
                         (1 + alpha_grid ** 2) * normal_cdf(-alpha_grid) - alpha_grid * normal_pdf(alpha_grid)))
    rho_se = np.max(rho_se)
    xhat = np.zeros((batch_size, K, M, P), dtype=complex)
    lamda = rho_se * d * np.ones((batch_size, K, M, P))
    iteration = np.zeros((batch_size))
    NMSE_watch = np.zeros((batch_size, iters))
    for sim in tqdm_notebook(range(batch_size), desc='simulation'):
        xmean = 0
        xvar = 0
        nvar = np.zeros(P)
        for p in range(P):
            for t in range(M):
                nvar[p] += np.linalg.norm(y[sim, :, t, p]) ** 2 / ((1 + snr0) * G)
                xvar += (np.linalg.norm(y[sim, :, t, p]) ** 2 - G * nvar[p]) / (
                            rho_se * d * np.linalg.norm(pilot[sim, :, :, p], ord='fro') ** 2)
        xvar = xvar / M / P
        nvar = nvar / M
        v = np.ones((K, M, P)) * xvar
        V = np.ones((G, M, P))
        Z = y[sim].copy()
        S = np.zeros((K, M, P))
        R = np.zeros((K, M, P), dtype=complex)
        m = np.zeros((K, M, P), dtype=complex)
        L = np.zeros((K, M, P))
        Vrs = np.zeros((K, M, P))
        for it in range(iters):
            x_pre = xhat[sim].copy()
            for p in range(P):
                V_new = (np.abs(pilot[sim, :, :, p]) ** 2).dot(v[:, :, p])
                Z_new = pilot[sim, :, :, p].dot(xhat[sim, :, :, p]) - (y[sim, :, :, p] - Z[:, :, p]) / (
                            nvar[p] + V[:, :, p]) * V_new
                V[:, :, p] = damp * V[:, :, p] + (1 - damp) * V_new
                Z[:, :, p] = damp * Z[:, :, p] + (1 - damp) * Z_new
                S[:, :, p] = 1 / (np.abs(pilot[sim, :, :, p]) ** 2).T.dot(1 / (nvar[p] + V[:, :, p]))
                R[:, :, p] = (np.conj(pilot[sim, :, :, p]).T.dot(
                    (y[sim, :, :, p] - Z[:, :, p]) / (nvar[p] + V[:, :, p]))) * S[:, :, p] + xhat[sim, :, :, p]
                L[:, :, p] = 0.5 * (np.log(S[:, :, p] / (S[:, :, p] + xvar)) + np.abs(R[:, :, p]) ** 2 / S[:, :, p] - (
                    np.abs(R[:, :, p] - xmean)) ** 2 / (S[:, :, p] + xvar))
                lamda[sim, :, :, p] /= lamda[sim, :, :, p] + (1 - lamda[sim, :, :, p]) * np.exp(-L[:, :, p])
                m[:, :, p] = (xvar * R[:, :, p] + xmean * S[:, :, p]) / (S[:, :, p] + xvar)
                Vrs[:, :, p] = xvar * S[:, :, p] / (xvar + S[:, :, p])
                xhat[sim, :, :, p] = lamda[sim, :, :, p] * m[:, :, p]
                v[:, :, p] = lamda[sim, :, :, p] * (np.abs(m[:, :, p]) ** 2 + Vrs[:, :, p]) - np.abs(
                    xhat[sim, :, :, p]) ** 2
                nvar[p] = np.sum(
                    np.abs(y[sim, :, :, p] - Z[:, :, p]) ** 2 / np.abs(1 + V[:, :, p] / nvar[p]) ** 2 + V[:, :, p] / (
                                1 + V[:, :, p] / nvar[p])) / (G * M)
            xmean = np.sum(lamda[sim] * m) / np.sum(lamda[sim])
            xvar = np.sum(lamda[sim] * (np.abs(xmean - m) ** 2 + Vrs)) / np.sum(lamda[sim])

            if P == 1:
                pi = lamda[sim]
                pi_update = np.sum(pi, axis=1) / M
                for k in range(K):
                    lamda[sim, k] = pi_update[k]
            else:
                pi = np.transpose(lamda[sim], (1, 2, 0))
                pi_update = np.sum(pi) / (M * P)
                for k in range(K):
                    lamda[sim, k] = pi_update[k]

            NMSE_watch[sim, it] = np.linalg.norm(x_pre - xhat[sim]) / np.linalg.norm(xhat[sim])
            if NMSE_watch[sim, it] < tol:
                iteration[sim] = it
                break
    return xhat, lamda, iteration, NMSE_watch


def lamda_aud(lamda, N_bs):
    batch_size, K, M, P = lamda.shape
    act_lam = np.zeros((batch_size, K))
    act_lam[np.sum(lamda > 0.5, axis=(2, 3)) > 0.9 * N_bs * P] = 1
    return act_lam


def channel_aud(H, N_bs):
    batch_size, K, M, P = H.shape
    act_ch = np.zeros((batch_size, K))
    p_th = np.max(np.abs(H), axis=(1, 2, 3), keepdims=True) * 0.01
    act_ch[np.sum((np.abs(H) > p_th), axis=(2, 3)) > 0.9 * N_bs * P] = 1
    return act_ch


def performance(det_act, true_act, det_H, true_H):
    _, K = det_act.shape
    Pe = np.mean(np.sum(np.abs(det_act - true_act), axis=1) / K)
    FA = np.mean(np.sum((det_act - true_act) == 1, axis=1) / K)
    PM = np.mean(np.sum((det_act - true_act) == -1, axis=1) / K)
    NMSE = np.linalg.norm(det_H - true_H) ** 2 / np.linalg.norm(true_H) ** 2
    return Pe, FA, PM, NMSE
