import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import os

from tqdm import trange
from torch.utils.data import DataLoader
from data.triplet_dataset import TripletDataset
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
        fea_dims = [data_X.data_mat.shape[1]] + fea_dims_sub

        self.data_X = data_X
        self.trn_index = trn_index
        self.Ytrn = data_Y[trn_index]
        self.Y = np.reshape(data_Y, [-1])
        self.pos_idx = np.intersect1d(np.argwhere(self.Y == 1.0), trn_index).flatten()
        self.neg_idx = np.intersect1d(np.argwhere(self.Y == 0.0), trn_index).flatten()

        self.model = DGP_RF_Embeddings(fea_dims, self.n_RF).cuda()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = ProbabilisticTripletLoss(alpha=self.alpha)


    def run_optimization(self, X_np, X_idx_np, Y_np, regul_const=1e-2):
        X = torch.tensor(X_np, dtype=torch.float32).cuda()
        X_idx = torch.tensor(X_idx_np, dtype=torch.long).cuda()
        Y = torch.tensor(Y_np, dtype=torch.float32).cuda()

        self.model.train()
        self.optimizer.zero_grad()

        est_means, est_vars = self.model(X, X_idx)
        loss = self.loss_fn(Y, est_means, est_vars, self.NpNm)
        reg_ = self.model.cal_regul()
        obj = loss + regul_const * reg_

        obj.backward()
        self.optimizer.step()

        return obj.item()
    
    def mark_subImgs(self, data_X, index_vec, sub_Ni=1, rep_num=1, flag_AllIns=False):
        # Randomly selects `sub_Ni` sub-instances per instance (unless flag_AllIns=True)
        Nis = np.hstack([data_X.Nis[idx] for idx in index_vec])
        set_indices = []

        for _ in range(rep_num):
            set_indices_sub = []
            for Ni in Nis:
                if not flag_AllIns:
                    selected = np.sort(np.random.choice(np.arange(Ni), size=min(Ni, sub_Ni), replace=False))
                else:
                    selected = np.arange(Ni)
                set_indices_sub.append(selected)
            set_indices.append(set_indices_sub)

        return set_indices
    
    def gen_input_fromList(self, data_X, index_vec, set_indices):
        X = []
        X_idx = []
        offset = 0

        for i, idx in enumerate(index_vec):
            instance = data_X.data_mat[idx]
            selected_rows = set_indices[i]
            X.append(instance[selected_rows])
            X_idx.extend([i] * len(selected_rows))

        X = np.vstack(X)
        X_idx = np.array(X_idx)
        return X, X_idx
    
    def predict(self, tst_index, sub_Ni=None, rep_num=1, flag_trndata=False):
        if sub_Ni is None:
            sub_Ni = self.sub_Ni

        if flag_trndata:
            data_set_ = self.data_X
        else:
            raise NotImplementedError("Non-training data prediction is not yet implemented")

        means_all, vars_all = [], []

        for idx in tst_index:
            set_indices = self.mark_subImgs(data_set_, [idx], sub_Ni=sub_Ni, rep_num=rep_num)[0]
            X, X_idx = self.gen_input_fromList(data_set_, [idx], set_indices)

            with torch.no_grad():
                X = torch.tensor(X, dtype=torch.float32).cuda()
                X_idx = torch.tensor(X_idx, dtype=torch.long).cuda()
                mean, var = self.model(X, X_idx)

            means_all.append(mean.cpu())
            vars_all.append(var.cpu())

        means_all = torch.cat(means_all, dim=0)
        vars_all = torch.cat(vars_all, dim=0)

        return means_all.numpy(), vars_all.numpy()



    def model_fit(self, save_path=None):
        # Khởi tạo TripletDataset & DataLoader
        dataset = TripletDataset(self.data_X, self.Y, self.pos_idx, self.neg_idx, self.sub_Ni)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        for epoch in trange(self.max_iter, desc="Training Epochs"):
            self.model.train()
            total_obj = 0.0

            for X, X_idx, Y in loader:
                # Đưa lên GPU
                X = X.cuda(non_blocking=True)
                X_idx = X_idx.cuda(non_blocking=True)
                Y = Y.cuda(non_blocking=True)

                # Forward & backward
                self.optimizer.zero_grad()
                est_means, est_vars = self.model(X, X_idx)
                loss = self.loss_fn(Y, est_means, est_vars, self.NpNm)
                reg = self.model.cal_regul()
                obj = loss + self.regul_const * reg
                obj.backward()
                self.optimizer.step()

                total_obj += obj.item()

            avg_obj = total_obj / len(loader)
            print(f"Epoch {epoch + 1}/{self.max_iter} - Loss: {avg_obj:.4f}")

        if save_path:
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

            