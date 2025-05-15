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
        NmulPnN: Normalization term
        """
        device = est_means.device
        idx_anchor = (y_true == -1)
        idx_pos = (y_true == 1)
        idx_neg = (y_true == 0)

        if idx_pos.sum() == 0 or idx_neg.sum() == 0:
            return torch.tensor(0.0, device=device)

        muA, muP, muN = est_means[idx_anchor], est_means[idx_pos], est_means[idx_neg]
        varA, varP, varN = est_vars[idx_anchor], est_vars[idx_pos], est_vars[idx_neg]

        T1 = ((muA - muP) ** 2 + varA + varP).sum(dim=1)
        T2 = ((muA - muN) ** 2 + varA + varN).sum(dim=1)
        T3 = (varA + varP).sum(dim=1).clamp(min=1e-8) + (varA + varN).sum(dim=1).clamp(min=1e-8)

        diff = T2 - T1
        sigma = torch.sqrt(T3)

        prob = Normal(loc=0.0, scale=sigma).cdf(diff - self.margin)
        loss = -torch.log(prob.clamp(min=1e-8)).mean()

        return self.alpha * loss / NmulPnN
