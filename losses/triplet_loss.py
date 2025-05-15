import torch
import torch.nn as nn
from torch.distributions import Normal

class ProbabilisticTripletLoss(nn.Module):
    def __init__(self, alpha=1.0, margin=1e-4):
        super().__init__()
        self.alpha = alpha
        self.margin = margin

    def forward(self, y_true, est_means, est_vars, NmulPnN):
        """
        y_true: [B] tensor with values [-1, 1, 0] (anchor, pos, neg)
        est_means: [B, D] mean embeddings
        est_vars: [B, D] variance embeddings
        NmulPnN: Normalization constant (float)
        """
        device = est_means.device
        idx_anchor = (y_true == -1)
        idx_pos = (y_true == 1)
        idx_neg = (y_true == 0)

        if idx_pos.sum() == 0 or idx_neg.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Anchor is assumed to be single (or first sample)
        muA = est_means[idx_anchor].squeeze(0)  # [D]
        varA = est_vars[idx_anchor].squeeze(0)  # [D]

        muP = est_means[idx_pos]  # [Np, D]
        varP = est_vars[idx_pos]  # [Np, D]

        muN = est_means[idx_neg]  # [Nn, D]
        varN = est_vars[idx_neg]  # [Nn, D]

        # Pairwise combinations (broadcasting over all pos-neg pairs)
        muP_ext = muP.unsqueeze(1)  # [Np, 1, D]
        varP_ext = varP.unsqueeze(1)
        muN_ext = muN.unsqueeze(0)  # [1, Nn, D]
        varN_ext = varN.unsqueeze(0)

        muA = muA.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        varA = varA.unsqueeze(0).unsqueeze(0)

        T1 = ((muA - muP_ext) ** 2 + varA + varP_ext).sum(dim=2)
        T2 = ((muA - muN_ext) ** 2 + varA + varN_ext).sum(dim=2)
        diff = T2 - T1  # shape: [Np, Nn]

        sigma2 = (2 * varA + varP_ext + varN_ext).sum(dim=2).clamp(min=1e-8)
        sigma = torch.sqrt(sigma2)

        probs = Normal(loc=0.0, scale=sigma).cdf(diff - self.margin)
        loss = -torch.log(probs.clamp(min=1e-8)).mean()

        return self.alpha * loss / NmulPnN
