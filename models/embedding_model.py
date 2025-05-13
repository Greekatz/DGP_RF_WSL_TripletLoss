import torch
import torch.nn as nn

from models.dgp_rff import DGP_RFF

class EmbeddingModel(nn.Module):
    def __init__(self, fea_dims, n_rff=100, sigma=1.0, prior_prec=10.0, normalize=True):
        """
        Wrapper for DGP_RFF to extract embeddings for Triplet Loss.

        Args:
            fea_dims (List[int]): Input and hidden dims including output.
            n_rff (int): Number of random Fourier features per layer.
            sigma (float): Lengthscale for RFF.
            prior_prec (float): Precision for KL divergence regularization.
            normalize (bool): L2 normalize final embeddings.
        """
        super(EmbeddingModel, self).__init__()

        self.dgp = DGP_RFF(
            fea_dims=fea_dims,
            n_rff=n_rff,
            sigma=sigma,
            prior_prec=prior_prec,
            normalize=normalize
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (B, input_dim)

        Returns:
            emb_mean: shape (B, output_dim)
            emb_var: shape (B, output_dim)
        """
        emb_mean, emb_var = self.dgp(x)
        return emb_mean, emb_var
        
    def kl_divergence(self):
        """
        Returns total KL divergence across DGP layers.
        """
        return self.dgp.kl_divergence()
                