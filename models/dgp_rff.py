import torch
import torch.nn as nn
from dgp_layer import DGPLayer  #

class DGP_RFF(nn.Module):
    def __init__(self, fea_dims, n_rff=100, prior_prec=1.0, normalize=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.normalize = normalize

        for i in range(len(fea_dims) - 1):
            self.layers.append(
                DGPLayer(
                    in_dim=fea_dims[i],
                    out_dim=fea_dims[i + 1],
                    n_rff=n_rff,
                    prior_prec=prior_prec,
                    is_output=(i == len(fea_dims) - 2)
                )
            )

    def forward(self, x, x_idx):
        x_var = None
        for layer in self.layers:
            x, x_var = layer(x, x_var)

        if self.normalize:
            x = nn.functional.normalize(x, p=2, dim=1)

        # Aggregate over indices
        return self.aggregate_embeddings(x, x_var, x_idx)

    def aggregate_embeddings(self, out_means, out_vars, x_idx):
        N = x_idx.max().item() + 1
        inv_vars = 1.0 / out_vars
        embed_vars = torch.zeros(N, out_means.shape[-1], device=out_means.device)
        embed_means = torch.zeros(N, out_means.shape[-1], device=out_means.device)

        for i in range(N):
            mask = (x_idx == i)
            embed_vars[i] = 1.0 / torch.sum(inv_vars[mask], dim=0)
            embed_means[i] = torch.sum(inv_vars[mask] * out_means[mask], dim=0)

        return embed_vars * embed_means, embed_vars

    def kl_divergence(self):
        return sum(layer.kl() for layer in self.layers)