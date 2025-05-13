import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class ProbabilisticTripletLoss(nn.Module):
    def __init__(self, margin=0.0, reduction='mean'):
        """
        Probabilistic Triplet Loss using Gaussian embeddings.

        Args:
            margin (float): Optional margin between positive and negative distances
            reduction (str): 'mean' or 'sum' (or 'none')
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, mu_a, var_a, mu_p, var_p, mu_n, var_n):
        """
        Args:
            mu_a, mu_p, mu_n: mean embeddings of shape (B, D)
            var_a, var_p, var_n: variance embeddings of shape (B, D)

        Returns:
            Scalar loss (float Tensor)
        """
        # Expected squared distances
        dist_ap = ((mu_a - mu_p)**2 + var_a + var_p).sum(dim=1)  # shape (B,)
        dist_an = ((mu_a - mu_n)**2 + var_a + var_n).sum(dim=1)

        # Distance difference: Dan - Dap - margin
        mu_diff = dist_an - dist_ap - self.margin

        # Variance of distance difference
        var_diff = 2 * (var_a**2 + var_p**2 + var_n**2).sum(dim=1)
        std_diff = torch.sqrt(torch.clamp(var_diff, min=1e-8))

        # Probability that Dap + margin < Dan
        normal = Normal(loc=mu_diff, scale=std_diff)
        log_prob = torch.log(normal.cdf(torch.zeros_like(mu_diff) + 1e-5))  # avoid log(0)
        loss = -log_prob

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
