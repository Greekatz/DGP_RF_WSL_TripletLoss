import torch.nn as nn
from models.dgp_rff import DGP_RFF


class EmbeddingModel(nn.Module):
    def __init__(self, fea_dims, n_rff=100, prior_prec=1.0):
        super().__init__()
        self.dgp = DGP_RFF(fea_dims, n_rff, prior_prec)

    def forward(self, x, x_idx):
        return self.dgp(x, x_idx)

    def kl_divergence(self):
        return self.dgp.kl_divergence()