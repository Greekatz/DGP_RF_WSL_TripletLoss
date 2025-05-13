import torch
import torch.nn as nn
from models.VBPLayer import VBPLinear  


class DGPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_rff, sigma=1.0, prior_prec=10.0, is_output=False):
        super(DGPLayer, self).__init__()

        self.rff_vb = VBPLinear(in_features=int(in_dim), out_features=n_rff, prior_prec=prior_prec, isoutput=True)
        self.weight_vb = VBPLinear(in_features=int(n_rff), out_features=out_dim, prior_prec=prior_prec, isoutput=is_output)

    def forward(self, input_mean, input_var=None):
        # First vb layer (like OmegaLayer in TF)
        if input_mean.dim() != 2:
            raise ValueError(f"Expected input_mean with 2 dims (B, in_dim), got shape {input_mean.shape}")
        phi_mean = self.rff_vb(input_mean)
        phi_var = self.rff_vb.var(input_mean, input_var)

        # Then projection layer
        output_mean = self.weight_vb(phi_mean)
        output_var = self.weight_vb.var(phi_mean, phi_var)

        return output_mean, output_var

    def calculate_kl(self):
        return self.rff_vb.KL() + self.weight_vb.KL()
