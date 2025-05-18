import torch
import torch.nn as nn
import torch.nn.functional as F

from models.VBPLayer import VBLayer

class DGP_RF_Embeddings(nn.Module):
    def __init__(self, fea_dims, num_RF):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(fea_dims) - 1):
            # Omega layer: non-linear transformation using VBLayer
            self.layers.append(VBLayer(num_RF, fea_dims[i], is_ReLUoutput=True))
            # Weight layer: linear projection using VBLayer
            self.layers.append(VBLayer(fea_dims[i + 1], num_RF))

    def forward(self, X, X_idx):
        inter_means = X
        inter_vars = torch.zeros_like(X)

        for layer in self.layers:
            inter_means, inter_vars = layer(inter_means, inter_vars)

        out_means = inter_means
        out_vars = torch.clamp(inter_vars, min=1e-8)

        # Uncertainty-aware pooling: weighted aggregation
        weighted = out_means / out_vars
        unique_ids = torch.unique(X_idx)
        embedd_means = []
        embedd_vars = []

        for uid in unique_ids:
            mask = (X_idx == uid)
            var_inv_sum = out_vars[mask].reciprocal().sum(dim=0) + 1e-8
            mean_sum = weighted[mask].sum(dim=0)
            var = 1.0 / var_inv_sum
            mean = mean_sum * var
            embedd_means.append(mean.unsqueeze(0))
            embedd_vars.append(var.unsqueeze(0))

        embedd_means = torch.cat(embedd_means, dim=0)
        embedd_vars = torch.cat(embedd_vars, dim=0)

        return embedd_means, embedd_vars

    def cal_regul(self):
        return sum(layer.calculate_kl() for layer in self.layers if hasattr(layer, "calculate_kl"))
