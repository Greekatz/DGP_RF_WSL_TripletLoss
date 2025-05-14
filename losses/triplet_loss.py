import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class ProbabilisticTripletLoss(nn.Module):
    def __init__(self, alpha=0.5, margin=0.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.eps = eps

    def forward(self, y_true, est_means, est_vars, NmulPnN):
        idx_pos = (y_true == 1).nonzero(as_tuple=False).squeeze()
        idx_neg = (y_true == 0).nonzero(as_tuple=False).squeeze()

        if idx_pos.numel() == 0 or idx_neg.numel() == 0:
            return torch.tensor(0.0, device=est_means.device)

        n_pos = idx_pos.numel()
        n_neg = idx_neg.numel()

        idx_pos_ex = idx_pos.repeat_interleave(n_neg)
        idx_neg_ex = idx_neg.repeat(n_pos)

        muA = est_means[0].unsqueeze(0)
        muP = est_means[idx_pos_ex]
        muN = est_means[idx_neg_ex]

        varA = est_vars[0].unsqueeze(0)
        varP = est_vars[idx_pos_ex]
        varN = est_vars[idx_neg_ex]

        probs = self.calculate_lik_prob(muA, muP, muN, varA, varP, varN)
        loss = -torch.sum(torch.log(probs + self.eps))

        const_ = NmulPnN / (self.alpha * float(n_pos * n_neg))
        return const_ * loss

    def calculate_lik_prob(self, muA, muP, muN, varA, varP, varN):
        muA2 = muA.pow(2)
        muP2 = muP.pow(2)
        muN2 = muN.pow(2)

        varP2 = varP.pow(2)
        varN2 = varN.pow(2)

        mu = torch.sum(muP2 + varP - muN2 - varN - 2 * muA * (muP - muN), dim=1)

        T1 = varP2 + 2 * muP2 * varP + 2 * (varA + muA2) * (varP + muP2) - 2 * muA2 * muP2 - 4 * muA * muP * varP
        T2 = varN2 + 2 * muN2 * varN + 2 * (varA + muA2) * (varN + muN2) - 2 * muA2 * muN2 - 4 * muA * muN * varN
        T3 = 4 * muP * muN * varA

        sigma = torch.sqrt(torch.clamp(torch.sum(2 * T1 + 2 * T2 - 2 * T3, dim=1), min=0.0))
        dist = Normal(loc=mu, scale=sigma + self.eps)
        return dist.cdf(torch.tensor(self.margin, device=mu.device))
    

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
