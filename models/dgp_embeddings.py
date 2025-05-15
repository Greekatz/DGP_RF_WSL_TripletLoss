import torch
import torch.nn as nn
from models.VBPLayer import VBLayer

class DGP_RF_Embeddings(nn.Module):
    def __init__(self, fea_dims, num_RF):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(fea_dims) - 1):
            self.layers.append(VBLayer(num_RF, fea_dims[i], is_ReLUoutput=True))  # Omega
            self.layers.append(VBLayer(fea_dims[i + 1], num_RF))  # Weight

    def forward(self, X, X_idx):
        inter_means, inter_vars = X, None
        

        for layer in self.layers:
            inter_means, inter_vars = layer(inter_means, inter_vars)

        embed_dim = inter_means.shape[1]
        var = inter_vars + 1e-8
        precision = 1.0 / var
        weighted = precision * inter_means

        unique_ids = torch.unique(X_idx, sorted=True)
        embed_dim = inter_means.shape[1]

        embedd_means = torch.zeros((len(unique_ids), embed_dim), device=X.device)
        embedd_vars = torch.zeros_like(embedd_means)

        for i, uid in enumerate(unique_ids):
            indices = (X_idx == uid).nonzero(as_tuple=True)[0]
            selected_weighted = weighted[indices]               # shape: [n_i, D]
            selected_precision = precision[indices]             # shape: [n_i, D]

            w_sum = selected_precision.sum(dim=0) + 1e-8         # shape: [D]
            mean_sum = selected_weighted.sum(dim=0)              # shape: [D]
            var_i = 1.0 / w_sum                                  # shape: [D]

            embedd_means[i] = (mean_sum * var_i).view(-1)       # ensure shape: [D]
            embedd_vars[i] = var_i.view(-1)                     # ensure shape: [D]

        return embedd_means, embedd_vars

    def cal_regul(self):
        return sum(layer.calculate_kl() for layer in self.layers)
