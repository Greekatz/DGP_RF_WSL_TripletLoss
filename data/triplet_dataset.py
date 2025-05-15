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
        pos_idx = np.random.choice(np.setdiff1d(self.pos_idx, anc_idx))
        neg_idx = np.random.choice(self.neg_idx)

        indices = [anc_idx, pos_idx, neg_idx]
        X_list, X_idx = [], []

        for i_sub, idx in enumerate(indices):
            vec = self.data_X.data_mat[idx]  # shape: (704,)
            X_list.append(vec[None, :])      # → shape: (1, 704)
            X_idx.append(i_sub)
            print("vec.shape:", vec.shape)


        X = np.vstack(X_list)               # → shape: (3, 704)
        X_idx = np.array(X_idx)             # [0, 1, 2]
        Y = np.array([-1, 1, 0], dtype=np.float32)

        return (
            torch.from_numpy(X).float(), 
            torch.tensor(X_idx, dtype=torch.long), 
            torch.from_numpy(Y)
        )
