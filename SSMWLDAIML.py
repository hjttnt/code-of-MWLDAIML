import torch
import numpy as np
import scipy
import measure

def mll_metrics(y_test, y_pred, y_score):
    scorce = {}

    hamming_loss = measure.hamming_loss_my(y_test, y_pred, y_score)
    coverage = measure.coverage(y_test, y_pred, y_score)
    ranking_loss = measure.ranking_loss(y_test, y_pred, y_score)
    one_error = measure.one_error(y_test, y_pred, y_score)
    average_precision = measure.average_precision(y_test, y_pred, y_score)
    subset_accuracy = measure.subset_accuracy(y_test, y_pred, y_score)
    accuracy = measure.accuracy(y_test, y_pred, y_score)
    precision = measure.precision(y_test, y_pred, y_score)
    recall = measure.recall(y_test, y_pred, y_score)
    f1 = measure.f1(y_test, y_pred, y_score)
    auc = measure.auc(y_test, y_pred, y_score)
    macro_averaging_accuracy = measure.macro_averaging_accuracy(y_test, y_pred, y_score)
    macro_averaging_precision = measure.macro_averaging_precision(y_test, y_pred, y_score)
    macro_averaging_recall = measure.macro_averaging_recall(y_test, y_pred, y_score)
    macro_averaging_f1 = measure.macro_averaging_f1(y_test, y_pred, y_score)
    macro_averaging_auc = measure.macro_averaging_auc(y_test, y_pred, y_score)
    micro_averaging_accuracy= measure.micro_averaging_accuracy(y_test, y_pred, y_score)
    micro_averaging_precision = measure.micro_averaging_precision(y_test, y_pred, y_score)
    micro_averaging_recall = measure.micro_averaging_recall(y_test, y_pred, y_score)
    micro_averaging_f1 = measure.micro_averaging_f1(y_test, y_pred, y_score)
    micro_averaging_auc = measure.micro_averaging_auc(y_test, y_pred, y_score)

    scorce["hamming_loss"] = hamming_loss
    scorce["coverage"] = coverage
    scorce["one_error"] = one_error
    scorce["average_precision"] = average_precision
    scorce["ranking_loss"] = ranking_loss
    scorce["subset_accuracy"] = subset_accuracy
    scorce["accuracy"] = accuracy
    scorce["precision"] = precision
    scorce["recall"] = recall
    scorce["f1"] = f1
    scorce["auc"] = auc
    scorce["macro_averaging_accuracy"] = macro_averaging_accuracy
    scorce["macro_averaging_precision"] = macro_averaging_precision
    scorce["macro_averaging_recall"] = macro_averaging_recall
    scorce["macro_averaging_f1"] = macro_averaging_f1
    scorce["macro_averaging_auc"] = macro_averaging_auc
    scorce["micro_averaging_accuracy"] = micro_averaging_accuracy
    scorce["micro_averaging_precision"] = micro_averaging_precision
    scorce["micro_averaging_recall"] = micro_averaging_recall
    scorce["micro_averaging_f1"] = micro_averaging_f1
    scorce["micro_averaging_auc"] = micro_averaging_auc
    return scorce

def _one(shape, device):
    return torch.randn(shape, dtype=torch.float64).to(device)

def lt_train_test(X, Y_train, Y_true,  L, Sw, Sb, G,  d, l, device, miu_max, miu1_max, miu2_max, miu3_max, miu4_max,
                  alpha, beta, lamda1, lamda2, lamda3, rou, rou1, rou2, rou3, rou4, yuzhi,
                  miu, miu1, miu2, miu3, miu4):
    (n, l) = Y_train.shape
    (m, l) = Y_true.shape
    Y_ext = torch.cat([Y_train, torch.zeros((m, l), device=device)], dim=0)
    W = _one((d, l), device) * 1
    K = _one((l, l), device) * 1
    Z = _one((l, l), device) * 1
    H = _one((l, l), device) * 1
    M = _one((d, l), device) * 1
    N = _one((d, l), device) * 1
    F = Y_ext
    mask_vec = torch.cat([
        torch.ones(n, dtype=torch.float64, device=device),
        torch.zeros(m, dtype=torch.float64, device=device)
    ])
    Mask = torch.diag(mask_vec)  # (n+m) x (n+m)
    gamma = 0
    Lambda1 = torch.zeros((d, l), dtype=torch.float64).to(device)
    Lambda2 = torch.zeros((d, l), dtype=torch.float64).to(device)
    Lambda3 = torch.zeros((l, l), dtype=torch.float64).to(device)
    Lambda4 = torch.zeros((n + m, l), dtype=torch.float64).to(device)
    step = 0
    while step <= 350:
        Wt = W
        Kt = K
        Zt = Z
        Ht = H
        Mt = M
        Nt = N
        K = torch.mm(torch.inverse(miu3 * torch.eye(l, dtype=torch.float64).cuda() + miu1 * torch.mm(W.T, W)),
                     (miu1 * torch.mm(W.T, M) - torch.mm(W.T, Lambda1) + miu3 * Z + Lambda3))
        U, S, V = torch.svd(K - 1 / miu3 * Lambda3)
        S_hat = torch.diag(torch.maximum(torch.zeros_like(S), S - lamda2 / miu3))
        Z = U @ S_hat @ V.T
        H = torch.mm(torch.inverse(2 * lamda3 * torch.eye(l, dtype=torch.float64).cuda() + miu2 * torch.mm(W.T, W)),
                     (miu2 * torch.mm(W.T, N) - torch.mm(W.T, Lambda2)))
        M = torch.mm(torch.inverse(2 * beta * Sw + miu1 * torch.eye(d, dtype=torch.float64).cuda()),
                     (miu1 * torch.mm(W, K) + Lambda1))
        N = torch.mm(torch.inverse(-2 * gamma * Sb + miu2 * torch.eye(d, dtype=torch.float64).cuda()),
                     (miu2 * torch.mm(W, H) + Lambda2))
        W = torch.from_numpy(scipy.linalg.solve_sylvester(
            (torch.mm(X.T, X) + 2 * alpha * X.T @ L @ X + 2 * lamda1 * torch.eye(d, dtype=torch.float64).cuda()).cpu().numpy(),
            (miu1 * torch.mm(K, K.T) + miu2 * torch.mm(H, H.T)).cpu().numpy(),
            (X.T @ F @ G + miu1 * torch.mm(M, K.T) + miu2 * torch.mm(N, H.T) - torch.mm(Lambda1, K.T) - torch.mm(
                Lambda2, H.T)).cpu().numpy())).to(torch.float64).to(device)
        C = X @ W @ G.T + miu4 * Mask @ Y_ext - Mask @ Lambda4
        for i in range(n + m):
            if i < n:
                F[i, :] = C[i, :] @ torch.inverse(torch.mm(G, G.T) + miu4 * torch.eye(l, dtype=torch.float64).cuda())
            else:
                F[i, :] = C[i, :] @ torch.inverse(torch.mm(G, G.T))
        gamma = miu * (1 - torch.trace(H.T @ W.T @ Sb @ W @ H))
        miu = min(rou * miu, miu_max)
        Lambda1 = Lambda1 + miu1 * (M - torch.mm(W, K))
        miu1 = min(rou1 * miu1, miu1_max)
        Lambda2 = Lambda2 + miu2 * (N - torch.mm(W, H))
        miu2 = min(rou2 * miu2, miu2_max)
        Lambda3 = Lambda3 + miu3 * (Z - K)
        miu3 = min(rou3 * miu3, miu3_max)
        Lambda4 = Lambda4 + miu4 * Mask @ (F - Y_ext)
        miu4 = min(rou4 * miu4, miu4_max)
        diff_K = torch.norm(Kt - K, p=torch.inf)
        diff_H = torch.norm(Ht - H, p=torch.inf)
        diff_M = torch.norm(Mt - M, p=torch.inf)
        diff_N = torch.norm(Nt - N, p=torch.inf)
        diff_W = torch.norm(Wt - W, p=torch.inf)
        diff_Z = torch.norm(Zt - Z, p=torch.inf)
        diff_WH_N = torch.norm(torch.mm(W, H) - N, p=np.inf)
        diff_WK_M = torch.norm(torch.mm(W, K) - M, p=np.inf)
        step += 1
        if  diff_WH_N < 1e-3 and diff_WK_M  < 1e-3 and diff_K  < 1e-3 and diff_Z  < 1e0 and diff_H  < 1e-3 and diff_M  < 1e-3 and diff_N  < 1e-3 and diff_W  < 1e-3:
            break
        if diff_WH_N > 1e2 or diff_WK_M  > 1e2 or diff_K  > 1e2 or diff_Z  > 1e2 or diff_H  > 1e2 or diff_M  > 1e2 or diff_N  > 1e2 or diff_W  > 1e2 or step > 350:
            break

    Y_score = F[n:, :].cpu().numpy()
    Y_pred = np.where(Y_score > yuzhi, 1, 0)
    scorce = mll_metrics(Y_true, Y_pred, Y_score)
    return Y_score, scorce

def getParameters(X, X_train, Y, n, d, l, k, delta, ksai, epsilo_1, epsilo_0, device):
    u_1 = torch.mm(Y.T, X_train) / Y.sum(axis=0).reshape(-1, 1).clamp(min=1e-10)  # 防止除以零
    u_1[Y.sum(axis=0) == 0] = 0  
    u_0 = torch.mm((1 - Y).T, X_train) / (1 - Y).sum(axis=0).reshape(-1, 1).clamp(
        min=1e-10)
    u_0[(1 - Y).sum(axis=0) == 0] = 0 

    D_1 = torch.cdist(X_train, u_1, p=2) ** 2  
    mask_1 = (Y == 1)
    delta_1 = torch.clamp(D_1, min=epsilo_1)  
    omega_1 = torch.where(mask_1, 1 / delta_1, torch.zeros_like(delta_1))
    omega_1_sum = omega_1.sum(dim=1, keepdim=True)
    P_1 = omega_1 / omega_1_sum
    P_1[~mask_1] = 0  

    D_0 = torch.cdist(X_train, u_0, p=2) ** 2 
    mask_0 = (Y == 0)
    delta_0 = torch.clamp(D_0, min=epsilo_0)
    omega_0 = torch.where(mask_0, 1 / delta_0, torch.zeros_like(delta_0))
    omega_0_sum = omega_0.sum(dim=1, keepdim=True)
    P_0 = omega_0 / omega_0_sum
    P_0[~mask_0] = 0 

    miu_1 = (P_1.T @ X_train) / P_1.sum(dim=0, keepdim=True).T
    miu_1.nan_to_num_(0) 
    miu_0 = (P_0.T @ X_train) / P_0.sum(dim=0, keepdim=True).T
    miu_0.nan_to_num_(0)
    Miu = ((P_1.T @ X_train).sum(dim=0) + (P_0.T @ X_train).sum(dim=0)) / torch.sum(P_1 + P_0)
    Miu.nan_to_num_(0)

    Sw = torch.zeros((d, d), dtype=torch.float64, device=device)  
    for j in range(l):
        diff_1 = X_train - miu_1[j]  
        diff_0 = X_train - miu_0[j]  
        Sw += 0.5 * ((P_1[:, j:j + 1] * diff_1).T @ diff_1 + (P_0[:, j:j + 1] * diff_0).T @ diff_0) 

    Sb = (P_1.sum(axis=0).reshape(-1, 1) * (miu_1 - Miu)).T @ (miu_1 - Miu) + (
            P_0.sum(axis=0).reshape(-1, 1) * (miu_0 - Miu)).T @ (miu_0 - Miu)

    dist_matrix = torch.cdist(X, X, p=2)  # (N, N)
    _A = torch.exp(-dist_matrix ** 2 / delta)
    J = _A.mean(dim=1)
    A_masked = _A.clone()
    A_masked[_A < J.unsqueeze(1)] = 0 
    _D = torch.diag(A_masked.sum(dim=1))
    L = _D - A_masked

    G = torch.eye(l, dtype=torch.float64).to(device)
    for i in range(l):
        if Y.sum(axis=0)[i] != 0:
            G[i][i] = k * (torch.abs(torch.log((Y.sum(axis=0)[i] / n) / ksai)) + 1)

    return Sw, Sb, L, G

def SSMWLDAIML(X_train, Y, X_test, Y_true,  miu_max, miu1_max, miu2_max, miu3_max, miu4_max,
            alpha, beta, lamda1, lamda2, lamda3, rou, rou1, rou2, rou3, rou4, miu, miu1, miu2, miu3, miu4, yuzhi):
    device = torch.device("cuda" if torch.cuda.is_available() else "")
    epsilo_1 = 0.001
    epsilo_0 = 0.001
    ksai = 0.5
    k = 1
    delta = 1.0
    X_train = torch.from_numpy(X_train).to(torch.float64).to(device)
    Y = torch.from_numpy(Y).to(torch.float64).to(device)
    X_test = torch.from_numpy(X_test).to(torch.float64).to(device)
    (n, d) = X_train.shape
    X = torch.cat([X_train, X_test], dim=0)
    l = Y.shape[1]
    Sw, Sb, L, G = getParameters(X, X_train, Y, n, d, l, k, delta, ksai, epsilo_1, epsilo_0, device)
    scorce = lt_train_test(X, Y, Y_true, L, Sw, Sb, G, d, l, device, miu_max, miu1_max, miu2_max, miu3_max, miu4_max,
                  alpha, beta, lamda1, lamda2, lamda3, rou, rou1, rou2, rou3, rou4, yuzhi,
                  miu, miu1, miu2, miu3, miu4)

    return scorce



