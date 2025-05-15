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

        var = inter_vars + 1e-8
        precision = 1.0 / var
        weighted = precision * inter_means

        unique_ids = torch.unique(X_idx, sorted=True)
        embed_dim = inter_means.size(1)

        embedd_means = torch.zeros((len(unique_ids), embed_dim), device=X.device)
        embedd_vars = torch.zeros_like(embedd_means)

        uid2idx = {uid.item(): i for i, uid in enumerate(unique_ids)}

        for uid in unique_ids:
            idx = uid2idx[uid.item()]
            mask = (X_idx == uid)
            w_sum = precision[mask].sum(dim=0) + 1e-8
            mean_sum = weighted[mask].sum(dim=0)
            var_i = 1.0 / w_sum

            embedd_means[idx] = mean_sum * var_i
            embedd_vars[idx] = var_i

        return embedd_means, embedd_vars


    def cal_regul(self):
        return sum(layer.calculate_kl() for layer in self.layers)
