import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VBPLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_prec=10.0, isoutput=False):
        super().__init__()
        print(f"[VBPLinear] in={in_features}, out={out_features}, type(in)={type(in_features)}")
        self.in_features = in_features
        self.out_features = out_features
        self.prior_prec = prior_prec
        self.isoutput = isoutput

        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.gamma = nn.Parameter(torch.Tensor(out_features)) if not isoutput else None

        self.normal = False  # <- must be before reset if reset uses it
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_features)
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.normal_(-9, 0.001)
        self.bias.data.zero_()
        if self.gamma is not None:
            self.gamma.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.mu_w, self.bias)

    def var(self, mean: torch.Tensor, var: torch.Tensor = None, C=100) -> torch.Tensor:
        if self.isoutput:
            # Output layer variance
            m2s2_w = self.mu_w**2 + self.logsig2_w.exp()
            term1 = F.linear(var, m2s2_w) if var is not None else 0
            term2 = F.linear(mean**2, self.logsig2_w.exp())
            return term1 + term2
        else:
            # Hidden layer: ReLU-like nonlinear propagation
            logits = F.linear(mean, self.mu_w, self.bias)  # shape (B, out_features)
            pZ = torch.sigmoid(C * logits)

            m2s2_w = self.mu_w**2 + self.logsig2_w.exp()
            term1 = F.linear(var, m2s2_w) if var is not None else 0
            term2 = F.linear(mean**2, self.logsig2_w.exp())
            varh = term1 + term2

            mean_sq = F.linear(mean, self.mu_w, self.bias)**2
            return pZ * varh + pZ * (1 - pZ) * mean_sq

    def KL(self, loguniform=False):
        if loguniform:
            # Optional alternative KL using log-uniform prior (used in VBP repo)
            k1 = 0.63576
            k2 = 1.87320
            k3 = 1.48695
            log_alpha = self.logsig2_w - 2 * torch.log(torch.abs(self.mu_w) + 1e-8)
            kl = -torch.sum(
                k1 * torch.sigmoid(k2 + k3 * log_alpha)
                - 0.5 * F.softplus(-log_alpha)
                - k1
            )
        else:
            logsig2 = torch.clamp(self.logsig2_w, -11, 11)
            kl = 0.5 * torch.sum(
                self.prior_prec * (self.mu_w**2 + logsig2.exp())
                - logsig2
                - 1
                - math.log(self.prior_prec)
            )
        return kl

    def __repr__(self):
        return f"VBPLinear({self.in_features} → {self.out_features}, output={self.isoutput})"
