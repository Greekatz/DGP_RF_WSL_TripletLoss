import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import math
from os import path
import copy
import itertools

# Helper functions to replace tf_commands
def mat_mul(x, y):
    return torch.matmul(x, y)

def square(x):
    return torch.square(x)

def exp(x):
    return torch.exp(x)

def sigmoid(x):
    return torch.sigmoid(x)

def multiply(x, y):
    return torch.mul(x, y)

def reduce_sum(x, axis=None):
    if axis is None:
        return torch.sum(x)
    return torch.sum(x, dim=axis)

def log(x):
    return torch.log(x)

def divide(x, y):
    return torch.div(x, y)

def vec_rowwise(x):
    return x.unsqueeze(0)

def vec_colwise(x):
    return x.unsqueeze(-1)

def vec_flat(x):
    return x.view(-1)

# Variational Bayesian Layer (equivalent to vb_layer)
class VBPLayer(nn.Module):
    def __init__(self, units, dim_input, is_ReLUoutput=False, use_bias=True, prior_prec=1.0):
        super(VBPLayer, self).__init__()
        self.units = units
        self.dim_input = dim_input
        self.is_ReLUoutput = is_ReLUoutput
        self.use_bias = use_bias
        self.prior_prec = prior_prec

        # Initialize weights
        self.w_mus = nn.Parameter(torch.Tensor(dim_input, units))
        self.w_logsig2 = nn.Parameter(torch.Tensor(dim_input, units))
        if use_bias:
            self.b = nn.Parameter(torch.Tensor(1, units))
        else:
            self.b = torch.zeros(1, units)

        if is_ReLUoutput:
            self.gamma = nn.Parameter(torch.Tensor(1, units))
        else:
            self.gamma = None

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w_mus)
        nn.init.xavier_normal_(self.w_logsig2)
        if self.use_bias:
            nn.init.zeros_(self.b)
        if self.is_ReLUoutput:
            nn.init.zeros_(self.gamma)

    def forward(self, in_means, in_vars=None):
        # Mean computation
        out_means = mat_mul(in_means, self.w_mus) + self.b

        if not self.is_ReLUoutput:
            # Variance computation
            m2s2_w = square(self.w_mus) + exp(self.w_logsig2)
            term1 = mat_mul(in_vars, m2s2_w) if in_vars is not None else 0
            term2 = mat_mul(square(in_means), exp(self.w_logsig2))
            out_vars = term1 + term2
        else:
            # ReLU output with scaling
            pZ = sigmoid(out_means)
            factor_ = exp(0.5 * self.gamma) * (2 / math.sqrt(self.units))
            out_means = multiply(multiply(pZ, out_means), factor_)

            # Variance
            m2s2_w = square(self.w_mus) + exp(self.w_logsig2)
            term1 = mat_mul(in_vars, m2s2_w) if in_vars is not None else 0
            term2 = mat_mul(square(in_means), exp(self.w_logsig2))
            term3 = square(mat_mul(in_means, self.w_mus) + self.b)
            out_vars = multiply(pZ, term1 + term2) + multiply(multiply(pZ, 1 - pZ), term3)
            out_vars = multiply(square(factor_), out_vars)

        return out_means, out_vars

    def calculate_kl(self):
        w_logsig2 = torch.clamp(self.w_logsig2, -11, 11)
        kl = 0.5 * reduce_sum(
            multiply(self.prior_prec, square(self.w_mus) + exp(w_logsig2)) - w_logsig2 - math.log(self.prior_prec)
        )
        return kl