import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def stable_softmax(X, dim=2):
    max_vec = torch.max(X, dim, keepdim=True)
    exp_X = torch.exp(X - max_vec[0])
    sum_exp_X = torch.sum(exp_X, dim, keepdim=True)
    X_hat = exp_X / sum_exp_X
    return X_hat


def compute_nll_loss(targets, y_hat, mask, M=20):
    epsilon = 1e-6
    split_sizes = [1] + [20] * 6
    y = torch.split(y_hat, split_sizes, dim=2)

    eos_logit = y[0].squeeze()
    log_mixture_weights = F.log_softmax(y[1], dim=2)

    mu_1 = y[2]
    mu_2 = y[3]

    logstd_1 = y[4]
    logstd_2 = y[5]

    rho = torch.tanh(y[6])

    log_constant = log_mixture_weights - math.log(2 * math.pi) - logstd_1 - \
        logstd_2 - 0.5 * torch.log(epsilon + 1 - rho.pow(2))

    x1 = targets[:, :, 1:2]
    x2 = targets[:, :, 2:]

    std_1 = torch.exp(logstd_1) + epsilon
    std_2 = torch.exp(logstd_2) + epsilon

    X1 = ((x1 - mu_1) / std_1).pow(2)
    X2 = ((x2 - mu_2) / std_2).pow(2)
    X1_X2 = 2 * rho * (x1 - mu_1) * (x2 - mu_2) / (std_1 * std_2)

    Z = X1 + X2 - X1_X2

    X = -Z / (2 * (epsilon + 1 - rho.pow(2)))

    log_sum_exp = torch.logsumexp(log_constant + X, 2)
    BCE = nn.BCEWithLogitsLoss(reduction='none')

    loss_t = -log_sum_exp + BCE(eos_logit, targets[:, :, 0])
    loss = torch.sum(loss_t * mask)

    return loss
