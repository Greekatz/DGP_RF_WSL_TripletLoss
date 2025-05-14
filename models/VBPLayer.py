import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VBLayer(nn.Module):
    def __init__(self, units, dim_input, is_ReLUoutput=False,
                 use_bias=True,
                 kernel_initializer=torch.nn.init.xavier_normal_,
                 bias_initializer=torch.nn.init.zeros_,
                 prior_prec=1.0):
        super(VBLayer, self).__init__()
        self.units = units
        self.d_input = dim_input
        self.is_ReLUoutput = is_ReLUoutput
        self.use_bias = use_bias
        self.prior_prec = prior_prec

        # Weight means and log variances
        self.w_mus = nn.Parameter(torch.empty(dim_input, units))
        self.w_logsig2 = nn.Parameter(torch.empty(dim_input, units))

        # Bias term
        self.b = nn.Parameter(torch.empty(1, units)) if use_bias else None

        # Gamma for ARD in ReLU output
        if is_ReLUoutput:
            self.gamma = nn.Parameter(torch.empty(1, units))

        # Initialize weights
        kernel_initializer(self.w_mus)
        kernel_initializer(self.w_logsig2)
        if use_bias:
            bias_initializer(self.b)
        if is_ReLUoutput:
            bias_initializer(self.gamma)

    def forward(self, in_means, in_vars=None):
        out_means = torch.matmul(in_means, self.w_mus)
        if self.use_bias:
            out_means += self.b

        if not self.is_ReLUoutput:
            m2s2_w = self.w_mus.pow(2) + torch.exp(self.w_logsig2)
            term1 = torch.matmul(in_vars, m2s2_w) if in_vars is not None else 0
            term2 = torch.matmul(in_means.pow(2), torch.exp(self.w_logsig2))
            out_vars = term1 + term2
        else:
            pZ = torch.sigmoid(out_means)
            factor = torch.exp(0.5 * self.gamma) * (2 / math.sqrt(self.units))
            out_means = pZ * out_means * factor

            if in_vars is not None:
                m2s2_w = self.w_mus.pow(2) + torch.exp(self.w_logsig2)
                term1 = torch.matmul(in_vars, m2s2_w)
            else:
                term1 = 0

            term2 = torch.matmul(in_means.pow(2), torch.exp(self.w_logsig2))
            term3 = (torch.matmul(in_means, self.w_mus) + self.b).pow(2)

            out_vars = pZ * (term1 + term2) + pZ * (1 - pZ) * term3
            out_vars = out_vars * factor.pow(2)

        return out_means, out_vars

    def calculate_kl(self):
        w_logsig2_clipped = torch.clamp(self.w_logsig2, min=-11.0, max=11.0)
        kl = 0.5 * torch.sum(
            self.prior_prec * (self.w_mus.pow(2) + torch.exp(w_logsig2_clipped)) -
            w_logsig2_clipped - torch.log(torch.tensor(self.prior_prec))
        )
        return kl
