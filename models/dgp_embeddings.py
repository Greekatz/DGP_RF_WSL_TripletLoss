import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from models.VBPLayer import VBLayer

class DGP_RF_Embeddings(nn.Module):
    def __init__(self, fea_dims, num_RF):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(fea_dims) - 1):
            # First: Omega layer with nonlinear output
            self.layers.append(VBLayer(num_RF, fea_dims[i], is_ReLUoutput=True))
            # Then: Linear weight projection layer
            self.layers.append(VBLayer(fea_dims[i + 1], num_RF))

    def forward(self, X, X_idx):
        inter_means = X
        inter_vars = torch.zeros_like(X)  # ensure safe variance flow from the start

        for layer in self.layers:
            inter_means, inter_vars = layer(inter_means, inter_vars)

        out_means, out_vars = inter_means, inter_vars

        # Clamp variance for numerical safety
        out_vars = torch.clamp(out_vars, min=1e-8)
        mat_tmp1 = 1.0 / out_vars
        weighted = mat_tmp1 * out_means

        unique_ids = torch.unique(X_idx)
        embedd_means = []
        embedd_vars = []

        for uid in unique_ids:
            mask = (X_idx == uid)
            w_sum = mat_tmp1[mask].sum(dim=0) + 1e-8
            mean_sum = weighted[mask].sum(dim=0)
            var = 1.0 / w_sum
            embedd_means.append((mean_sum * var).unsqueeze(0))
            embedd_vars.append(var.unsqueeze(0))

        embedd_means = torch.cat(embedd_means, dim=0)
        embedd_vars = torch.cat(embedd_vars, dim=0)

        # Optional: L2-normalize embeddings (if used for cosine similarity)
        # embedd_means = F.normalize(embedd_means, p=2, dim=1)

        return embedd_means, embedd_vars

    def cal_regul(self):
        # Sum of KL-divergences from all VB layers
        mr_KLsum = 0.0
        for layer in self.layers:
            if hasattr(layer, "calculate_kl"):
                mr_KLsum += layer.calculate_kl()
        return mr_KLsum
