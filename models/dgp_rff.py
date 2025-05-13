import torch
import torch.nn as nn
from dgp_layer import DGPLayer  # Adjusted import if in same directory

class DGP_RFF(nn.Module):
    def __init__(self, fea_dims, n_rff=100, sigma=1.0, prior_prec=10.0, normalize=False):
        super(DGP_RFF, self).__init__()
        if not isinstance(fea_dims, list) or len(fea_dims) < 2:
            raise ValueError("fea_dims must be a list with at least two dimensions")
        self.normalize = normalize  # Added normalize parameter
        self.layers = nn.ModuleList()
        for i in range(len(fea_dims) - 1):
            self.layers.append(
                DGPLayer(
                    in_dim=fea_dims[i],
                    out_dim=fea_dims[i + 1],
                    n_rff=n_rff,
                    sigma=sigma,
                    prior_prec=prior_prec,
                    is_output=(i == len(fea_dims) - 2)
                )
            )

    def forward(self, x, x_var=None):
        for layer in self.layers:
            x, x_var = layer(x, x_var)
        if self.normalize:
            x = nn.functional.normalize(x, p=2, dim=1)
        return x, x_var

    def kl_divergence(self):
        return sum(layer.calculate_kl() for layer in self.layers)