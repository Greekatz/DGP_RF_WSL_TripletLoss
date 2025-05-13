import torch
import torch.nn as nn
from models.omega_layer import OmegaLayer
from models.VBPLayer import VBPLinear  


class DGPLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_rff: int,
        sigma: float = 1.0,
        prior_prec: float = 10.0,
        is_output: bool = False
    ):
        """
        One layer of a DGP with Random Fourier Features and Variational Bayesian Projection.

        Args:
            in_dim (int): Input feature dimension.
            out_dim (int): Output feature dimension.
            n_rff (int): Number of random Fourier features.
            sigma (float): Length scale for the RBF kernel.
            prior_prec (float): Prior precision for KL divergence.
            is_output (bool): Whether this is the final output layer.
        """
        super(DGPLayer, self).__init__()

        self.omega = OmegaLayer(in_dim, n_rff, sigma=sigma)
        self.linear = VBPLinear(in_features=n_rff, out_features=out_dim, prior_prec=prior_prec, isoutput=is_output)
        self.is_output = is_output

    def forward(self, input_mean: torch.Tensor, input_var: torch.Tensor = None):
        """
        Forward pass with variance propagation.

        Args:
            input_mean (Tensor): shape (B, in_dim), mean of the input.
            input_var (Tensor): shape (B, in_dim), optional variance of the input.

        Returns:
            output_mean (Tensor): shape (B, out_dim)
            output_var (Tensor): shape (B, out_dim)
        """
        # Apply RFF transformation
        phi_mean = self.omega(input_mean)

        # Propagate through variational linear layer
        output_mean = self.linear(phi_mean)
        output_var = self.linear.var(phi_mean, input_var)

        return output_mean, output_var

    def calculate_kl(self) -> torch.Tensor:
        """
        Compute KL divergence for this layer's weight posterior.

        Returns:
            kl (Tensor): KL divergence scalar
        """
        return self.linear.KL()
