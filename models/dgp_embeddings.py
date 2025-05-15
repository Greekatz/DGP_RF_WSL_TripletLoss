import torch
import torch.nn as nn
from models.VBPLayer import VBLinear

class DGP_RF_Embedding(nn.Module):
    def __init__(self, fea_dims, n_rff):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(fea_dims) - 1):
            self.layers.append(VBLinear(fea_dims[i], n_rff, relu_output=True))
            self.layers.append(VBLinear(n_rff, fea_dims[i + 1]))

    def forward(self, X, X_idx):
        means, vars = X, None
        for layer in self.layers:
            means, vars = layer(means, vars)

        inv_vars = 1 / vars
        summed_inv = torch.zeros_like(inv_vars).index_add(0, X_idx, inv_vars)
        summed_weighted = torch.zeros_like(means).index_add(0, X_idx, means * inv_vars)

        emb_vars = 1 / summed_inv
        emb_means = emb_vars * summed_weighted
        return emb_means, emb_vars

    def regularization(self):
        return sum(layer.kl_divergence() for layer in self.layers if isinstance(layer, VBLinear))