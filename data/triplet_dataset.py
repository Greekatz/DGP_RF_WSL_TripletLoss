import torch
import numpy as np
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, data_X, Y, pos_idx, neg_idx, sub_Ni):
        self.data_X = data_X
        self.Y = Y
        self.pos_idx = pos_idx
        self.neg_idx = neg_idx
        self.sub_Ni = sub_Ni

    def __len__(self):
        return len(self.pos_idx)

    def __getitem__(self, i):
        anc_idx = self.pos_idx[i]
        pos_pool = self.pos_idx[self.pos_idx != anc_idx]
        pos_sample = np.random.choice(pos_pool)
        neg_sample = np.random.choice(self.neg_idx)

        index_vec = [anc_idx, pos_sample, neg_sample]
        X_list, X_idx = [], []

        for i_sub, idx in enumerate(index_vec):
            vec = self.data_X.data_mat[idx]         # shape: (D,)
            if vec.ndim == 1:
                vec = vec[None, :]                  # shape: (1, D)
            X_list.append(vec)
            X_idx.extend([i_sub] * vec.shape[0])

        X = np.vstack(X_list)                       # shape: (3, D)
        X_idx = np.array(X_idx)                     # shape: (3,)
        Y = np.array([-1, 1, 0], dtype=np.float32)

        return (
            torch.from_numpy(X).float(),
            torch.tensor(X_idx, dtype=torch.long),
            torch.from_numpy(Y)
        )