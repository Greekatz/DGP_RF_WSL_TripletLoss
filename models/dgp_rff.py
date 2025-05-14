import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import os

from dgp_embeddings import DGP_RF_Embeddings
from losses.triplet_loss import ProbabilisticTripletLoss

class DGP_RF:
    def __init__(self, data_X, data_Y, trn_index, setting, str_filepath=None):
        N_pos = np.sum(data_Y[trn_index] == 1)
        self.NpNm = 0.5 * ((N_pos * (N_pos - 1)) * np.sum(data_Y[trn_index] == 0))

        self.max_iter = setting.max_iter
        self.iter_print = setting.iter_print
        self.sub_Ni = setting.sub_Ni
        self.batch_size = setting.batch_size
        self.ker_type = setting.ker_type
        self.n_RF = setting.n_RF
        self.regul_const = float(setting.regul_const)
        self.alpha = setting.alpha

        fea_dims_sub = [100] * setting.n_layers
        fea_dims = [data_X.data_mat[0].shape[1]] + fea_dims_sub

        self.data_X = data_X
        self.trn_index = trn_index
        self.Ytrn = data_Y[trn_index]
        self.Y = np.reshape(data_Y, [-1])
        self.pos_idx = np.intersect1d(np.argwhere(self.Y == 1.0), trn_index).flatten()
        self.neg_idx = np.intersect1d(np.argwhere(self.Y == 0.0), trn_index).flatten()

        self.model = DGP_RF_Embeddings(fea_dims, self.n_RF).cuda()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

        if str_filepath is None:
            self.model_fit()
        else:
            if os.path.exists(str_filepath + '.pt'):
                self.model.load_state_dict(torch.load(str_filepath + '.pt'))
                print(f"Loaded trained model from {str_filepath}")
            else:
                self.model_fit()
                torch.save(self.model.state_dict(), str_filepath + '.pt')

    def run_optimization(self, X_np, X_idx_np, Y_np, regul_const=1e-2):
        X = torch.tensor(X_np, dtype=torch.float32).cuda()
        X_idx = torch.tensor(X_idx_np, dtype=torch.long).cuda()
        Y = torch.tensor(Y_np, dtype=torch.float32).cuda()

        self.model.train()
        self.optimizer.zero_grad()

        est_means, est_vars = self.model(X, X_idx)
        loss = ProbabilisticTripletLoss(Y, est_means, est_vars, self.NpNm, self.alpha)
        reg_ = self.model.cal_regul()
        obj = loss + regul_const * reg_

        obj.backward()
        self.optimizer.step()

        return obj.item()
