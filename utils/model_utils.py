import torch
import torch.nn.functional as F
import math


def stable_softmax(X):
    max_vec = torch.max(X, 2, keepdim=True)
    exp_X = torch.exp(X - max_vec[0])
    sum_exp_X = torch.sum(exp_X, 2, keepdim=True)
    X_hat = exp_X / sum_exp_X
    return X_hat


def logSumExp(X, constant):
    """
       Numerically stable log sum exponential
    """
    max_vec = torch.max(X, 2, keepdim=True)
    exp_X = torch.exp(X - max_vec[0])
    exp_X = exp_X * consatnt
    log_sum = torch.log(torch.sum(exp_X, 2, keepdim=True)) + max_vec
    return log_sum.squeeze()


def compute_unconditional_loss(targets, y_hat, mask, M=20):
    split_sizes = [1] + [20] * 6
    y = torch.split(y_hat, split_sizes, dim=2)
    eos_prob = F.sigmoid(y[0])
    mixture_weights = stable_softmax(y[1])
    mu_1 = y[2]
    mu_2 = y[3]
    std_1 = torch.exp(y[4])
    std_2 = torch.exp(y[5])
    correlations = F.tanh(y[6])
    constant = mixture_weights / (2 * math.pi * std_1 * std_2 * torch.sqrt(1 - correlations.pow(2)))
    x1 = targets[:, :, 1:2]
    x2 = targets[:, :, 2:]
    X1 = ((x1 - mu_1) / std_1).pow(2)
    X2 = ((x2 - mu_2) / std_2).pow(2)
    X1_X2 = 2 * correlations * (x1 - mu_1) * (x2 - mu_2) / (std_1 * std_2)
    Z = X1 + X2 - X1_X2
    X = -Z / (2 * (1 - correlations.pow(2)))
    log_sum_exp = logSumExp(X, constant)
    loss_t = -log_sum_exp - torch.log(eos_prob * targets[:, :, 0] + (1 - eos_prob) * (1 - targets[:, :, 0]))
    loss = torch.sum(loss_t * mask)
    return loss
