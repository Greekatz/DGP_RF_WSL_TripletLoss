import torch
import torch.nn as nn
import math

class OmegaLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_type='rbf', sigma=1.0):
        """
        Args:
            in_dim (int): input feature size (d)
            out_dim (int): number of random features (D)
            kernel_type (str): currently supports 'rbf' only
            sigma (float): length scale for RBF kernel
        """
        super(OmegaLayer, self).__init__()
        assert kernel_type == 'rbf', "Only 'rbf' kernel is currently supported."

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma

        # Sample omega from N(0, 1/sigma^2)
        self.omega = nn.Parameter(
            torch.randn(out_dim, in_dim) / sigma, requires_grad=False
        )

        # Sample random bias uniformly from [0, 2π]
        self.bias = nn.Parameter(
            2 * math.pi * torch.rand(out_dim), requires_grad=False
        )

        self.scale = math.sqrt(2.0 / out_dim)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, in_dim)
        Returns:
            Random Fourier features of shape (batch_size, out_dim)
        """
        projection = x @ self.omega.T + self.bias  # shape: (batch_size, out_dim)
        return self.scale * torch.cos(projection)
