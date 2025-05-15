import numpy as np
import torch
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, data_X, Y, pos_idx, neg_idx, sub_Ni):
        self.data_X = data_X
        self.Y = Y
        self.pos_idx = pos_idx
        self.neg_idx = neg_idx
        self.sub_Ni = sub_Ni

        self.data_X.data_mat = [
            torch.tensor(ins, dtype=torch.float32) if not isinstance(ins, torch.Tensor) else ins
            for ins in self.data_X.data_mat
        ]

    def __len__(self):
        return len(self.pos_idx)

    def __getitem__(self, i):
        anc_idx = self.pos_idx[i]

        # Chọn positive sample khác anchor
        pos_pool = self.pos_idx[self.pos_idx != anc_idx]
        pos_sample = np.random.choice(pos_pool)
        neg_sample = np.random.choice(self.neg_idx)

        index_vec = [anc_idx, pos_sample, neg_sample]
        X_list, X_idx = [], []

        for i_sub, idx in enumerate(index_vec):
            instance = self.data_X.data_mat[idx]  # shape: [Ni, D]
            Ni = self.data_X.Nis[idx]

            # Nếu Ni <= sub_Ni, lấy tất cả
            if Ni <= self.sub_Ni:
                selected_rows = torch.arange(Ni)
            else:
                selected_rows = torch.randperm(Ni)[:self.sub_Ni]

            X_list.append(instance[selected_rows])
            X_idx.extend([i_sub] * len(selected_rows))

        X = torch.cat(X_list, dim=0)  # shape: [*, D]
        X_idx = torch.tensor(X_idx, dtype=torch.long)
        Y = torch.tensor([-1.0, 1.0, 0.0], dtype=torch.float32)

        return X, X_idx, Y
