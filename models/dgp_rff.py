import torch
import torch.nn as nn
from models.dgp_layer import DGPLayer


class DGP_RFF(nn.Module):
    def __init__(
        self,
        fea_dims,            # e.g., [128, 256, 256, 64]
        n_rff=100,
        sigma=1.0,
        prior_prec=10.0,
    ):
        """
        Deep Gaussian Process with Random Fourier Features.

        Args:
            fea_dims (List[int]): Dimensions for each layer including input & output.
            n_rff (int): Number of random Fourier features per layer.
            sigma (float): Lengthscale for RBF kernel.
            prior_prec (float): Prior precision (λ) for KL divergence.
            normalize (bool): Whether to L2-normalize final mean embeddings.
        """
        super(DGP_RFF, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(len(fea_dims) - 1):
            self.layers.append(
                DGPLayer(
                    in_dim=fea_dims[i],
                    out_dim=fea_dims[i + 1],
                    n_rff=n_rff,
                    sigma=sigma,
                    prior_prec=prior_prec,
                    is_output=(i == len(fea_dims) - 2)  # last layer
                )
            )

    def forward(self, x, x_var=None):
        """
        Forward pass through all DGP layers.

        Args:
            x (Tensor): Input mean, shape (B, in_dim)
            x_var (Tensor): Input variance (optional), shape (B, in_dim)

        Returns:
            final_mean: shape (B, out_dim)
            final_var: shape (B, out_dim)
        """
        for layer in self.layers:
            x, x_var = layer(x, x_var)

        if self.normalize:
            x = nn.functional.normalize(x, p=2, dim=1)

        return x, x_var

    def kl_divergence(self):
        """
        Sum KL divergences from all layers.

        Returns:
            total_kl (Tensor): KL divergence scalar
        """
        return sum(layer.calculate_kl() for layer in self.layers)
