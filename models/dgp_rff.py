import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import os

from torch_scatter import scatter_sum
from tqdm import trange
from models.dgp_embeddings import DGP_RF_Embeddings
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


    def model_fit(self):
        iters_Pos = len(self.pos_idx)
        n_pos = int(max(round(self.batch_size / 2), 1))
        n_neg = n_pos

        for epoch in trange(self.max_iter, desc="Training Epochs"):
            eta = 1.0 if epoch < 10 else 1.0
            total_obj = 0.0

            for i in range(iters_Pos):
                anc_idx = self.pos_idx[i]

                pos_idx = np.random.choice(
                    np.setdiff1d(self.pos_idx, anc_idx),
                    min(n_pos, len(self.pos_idx) - 1),
                    replace=False
                )
                pos_idx = np.concatenate(([anc_idx], pos_idx))

                neg_idx = np.random.choice(
                    self.neg_idx,
                    min(n_neg, len(self.neg_idx)),
                    replace=False
                )

                index_vec = np.concatenate((pos_idx, neg_idx))
                set_indices = self.mark_subImgs(self.data_X, index_vec, sub_Ni=self.sub_Ni)

                X, X_idx = self.gen_input_fromList(self.data_X, index_vec, set_indices[0])
                Y = self.Y[index_vec]
                Y[0] = -1

                total_obj += self.run_optimization(X, X_idx, Y, eta * self.regul_const)

            avg_obj = total_obj / iters_Pos
            print(f"Epoch {epoch + 1}/{self.max_iter} - Loss: {avg_obj:.4f}")

            if self.iter_print and (epoch == self.max_iter - 1):
                out_means, _ = self.predict(self.trn_index, sub_Ni=self.sub_Ni, rep_num=1, flag_trndata=True)
                auc_val = roc_auc_score(self.Ytrn, out_means)
                print(f"  Train AUC = {auc_val:.4f}")