import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class ProbabilisticTripletLoss(nn.Module):
    def __init__(self, margin=0.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, mu_a, var_a, mu_p, var_p, mu_n, var_n):
        dist_ap = ((mu_a - mu_p)**2 + var_a + var_p).sum(dim=1)
        dist_an = ((mu_a - mu_n)**2 + var_a + var_n).sum(dim=1)

        mu_diff = dist_an - dist_ap

        # variance of Dan - Dap
        var_diff = 2 * (var_a + var_p + var_n).sum(dim=1)
        std_diff = torch.sqrt(torch.clamp(var_diff, min=1e-8))

        normal = Normal(loc=mu_diff, scale=std_diff)
        cdf_val = normal.cdf(torch.tensor(self.margin, device=mu_diff.device))
        log_prob = torch.log(torch.clamp(cdf_val, min=1e-6))

        loss = -log_prob
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
