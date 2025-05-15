import torch
import torch.nn as nn
import math


class VBLinear(nn.Module):
    def __init__(self, in_features, out_features, relu_output=False):
        super(VBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.relu_output = relu_output

        self.w_mu = nn.Parameter(torch.empty(in_features, out_features).normal_(0, 0.1))
        self.w_logsig2 = nn.Parameter(torch.empty(in_features, out_features).normal_(-5, 0.1))
        self.b = nn.Parameter(torch.zeros(out_features))

        if relu_output:
            self.gamma = nn.Parameter(torch.zeros(out_features))

        self.prior_prec = 1.0

    def forward(self, mean_in, var_in):
        mean_out = mean_in @ self.w_mu + self.b

        if not self.relu_output:
            m2s2_w = self.w_mu**2 + torch.exp(self.w_logsig2)
            var_out = var_in @ m2s2_w + (mean_in**2) @ torch.exp(self.w_logsig2)
        else:
            pZ = torch.sigmoid(mean_out)
            factor = torch.exp(0.5 * self.gamma) * (2 / np.sqrt(self.out_features))
            mean_out = pZ * mean_out * factor

            m2s2_w = self.w_mu**2 + torch.exp(self.w_logsig2)
            term1 = var_in @ m2s2_w if var_in is not None else 0
            term2 = (mean_in**2) @ torch.exp(self.w_logsig2)
            term3 = (mean_in @ self.w_mu + self.b)**2
            var_out = pZ * (term1 + term2) + pZ * (1 - pZ) * term3
            var_out *= factor**2

        return mean_out, var_out

    def kl_divergence(self):
        logsig2 = torch.clamp(self.w_logsig2, -11, 11)
        kl = 0.5 * torch.sum(
            self.prior_prec * (self.w_mu**2 + torch.exp(logsig2)) - logsig2 - np.log(self.prior_prec)
        )
        return kl
