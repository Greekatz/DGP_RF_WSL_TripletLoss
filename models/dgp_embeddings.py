import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch_scatter import scatter_sum

from models.VBPLayer import VBLayer

class DGP_RF_Embeddings(nn.Module):
    def __init__(self, fea_dims, num_RF):
        super(DGP_RF_Embeddings, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(fea_dims) - 1):
            # Omega layer
            self.layers.append(VBLayer(num_RF, fea_dims[i], is_ReLUoutput=True))
            # Weight layer
            self.layers.append(VBLayer(fea_dims[i + 1], num_RF))

    def forward(self, X, X_idx):
        inter_means, inter_vars = X, None

        for layer in self.layers:
            inter_means, inter_vars = layer(inter_means, inter_vars)

        out_means, out_vars = inter_means, inter_vars
        mat_tmp1 = 1.0 / (out_vars + 1e-8)  # prevent div-by-zero
        weighted = mat_tmp1 * out_means

        unique_ids = torch.unique(X_idx)
        embedd_means, embedd_vars = [], []

        for uid in unique_ids:
            mask = (X_idx == uid)
            w_sum = mat_tmp1[mask].sum(dim=0) + 1e-8
            mean = weighted[mask].sum(dim=0)
            var = 1.0 / w_sum
            embedd_means.append((mean * var).unsqueeze(0))
            embedd_vars.append(var.unsqueeze(0))

        return torch.cat(embedd_means, dim=0), torch.cat(embedd_vars, dim=0)


    def cal_regul(self):
        mr_KLsum = 0.0
        for layer in self.layers:
            mr_KLsum += layer.calculate_kl()
        return mr_KLsum
