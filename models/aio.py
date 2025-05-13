import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import math
from os import path
import copy
import itertools

# Helper functions to replace tf_commands
def mat_mul(x, y):
    return torch.matmul(x, y)

def square(x):
    return torch.square(x)

def exp(x):
    return torch.exp(x)

def sigmoid(x):
    return torch.sigmoid(x)

def multiply(x, y):
    return torch.mul(x, y)

def reduce_sum(x, axis=None):
    if axis is None:
        return torch.sum(x)
    return torch.sum(x, dim=axis)

def log(x):
    return torch.log(x)

def divide(x, y):
    return torch.div(x, y)

def vec_rowwise(x):
    return x.unsqueeze(0)

def vec_colwise(x):
    return x.unsqueeze(-1)

def vec_flat(x):
    return x.view(-1)

# Variational Bayesian Layer (equivalent to vb_layer)
class VBPLayer(nn.Module):
    def __init__(self, units, dim_input, is_ReLUoutput=False, use_bias=True, prior_prec=1.0):
        super(VBPLayer, self).__init__()
        self.units = units
        self.dim_input = dim_input
        self.is_ReLUoutput = is_ReLUoutput
        self.use_bias = use_bias
        self.prior_prec = prior_prec

        # Initialize weights
        self.w_mus = nn.Parameter(torch.Tensor(dim_input, units))
        self.w_logsig2 = nn.Parameter(torch.Tensor(dim_input, units))
        if use_bias:
            self.b = nn.Parameter(torch.Tensor(1, units))
        else:
            self.b = torch.zeros(1, units)

        if is_ReLUoutput:
            self.gamma = nn.Parameter(torch.Tensor(1, units))
        else:
            self.gamma = None

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w_mus)
        nn.init.xavier_normal_(self.w_logsig2)
        if self.use_bias:
            nn.init.zeros_(self.b)
        if self.is_ReLUoutput:
            nn.init.zeros_(self.gamma)

    def forward(self, in_means, in_vars=None):
        # Mean computation
        out_means = mat_mul(in_means, self.w_mus) + self.b

        if not self.is_ReLUoutput:
            # Variance computation
            m2s2_w = square(self.w_mus) + exp(self.w_logsig2)
            term1 = mat_mul(in_vars, m2s2_w) if in_vars is not None else 0
            term2 = mat_mul(square(in_means), exp(self.w_logsig2))
            out_vars = term1 + term2
        else:
            # ReLU output with scaling
            pZ = sigmoid(out_means)
            factor_ = exp(0.5 * self.gamma) * (2 / math.sqrt(self.units))
            out_means = multiply(multiply(pZ, out_means), factor_)

            # Variance
            m2s2_w = square(self.w_mus) + exp(self.w_logsig2)
            term1 = mat_mul(in_vars, m2s2_w) if in_vars is not None else 0
            term2 = mat_mul(square(in_means), exp(self.w_logsig2))
            term3 = square(mat_mul(in_means, self.w_mus) + self.b)
            out_vars = multiply(pZ, term1 + term2) + multiply(multiply(pZ, 1 - pZ), term3)
            out_vars = multiply(square(factor_), out_vars)

        return out_means, out_vars

    def calculate_kl(self):
        w_logsig2 = torch.clamp(self.w_logsig2, -11, 11)
        kl = 0.5 * reduce_sum(
            multiply(self.prior_prec, square(self.w_mus) + exp(w_logsig2)) - w_logsig2 - math.log(self.prior_prec)
        )
        return kl

# DGP-RF Embeddings Model
class DGP_RF_Embeddings(nn.Module):
    def __init__(self, fea_dims, num_RF):
        super(DGP_RF_Embeddings, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(len(fea_dims) - 1):
            # Omega layer (ReLU output)
            self.layers.append(
                VBPLayer(units=num_RF, dim_input=fea_dims[i], is_ReLUoutput=True)
            )
            # Weight layer
            self.layers.append(
                VBPLayer(units=fea_dims[i + 1], dim_input=num_RF, is_ReLUoutput=False)
            )

    def forward(self, x, x_idx):
        # Forward pass through DGP layers
        inter_means, inter_vars = x, None
        for layer in self.layers:
            inter_means, inter_vars = layer(inter_means, inter_vars)

        # Aggregate embeddings to image level
        embedd_vars, embedd_means = self.aggregate_embeddings(inter_means, inter_vars, x_idx)
        return multiply(embedd_vars, embedd_means), embedd_vars

    def aggregate_embeddings(self, out_means, out_vars, x_idx):
        # Compute inverse variances
        mat_tmp1 = divide(1, out_vars) if out_vars is not None else torch.ones_like(out_means)

        # Segment sum for means and variances
        unique_idx, inverse_idx = torch.unique(x_idx, return_inverse=True)
        embedd_vars = torch.zeros(len(unique_idx), out_means.size(-1), device=out_means.device)
        embedd_means = torch.zeros(len(unique_idx), out_means.size(-1), device=out_means.device)

        for i, idx in enumerate(unique_idx):
            mask = (inverse_idx == i)
            embedd_vars[i] = 1 / torch.sum(mat_tmp1[mask], dim=0)
            embedd_means[i] = torch.sum(multiply(mat_tmp1[mask], out_means[mask]), dim=0)

        return embedd_vars, embedd_means

    def cal_regul(self):
        return sum(layer.calculate_kl() for layer in self.layers)

# Probabilistic Triplet Loss
def prob_triplet_loss(y_true, est_means, est_mvars, NmulPnN, alpha=0.5):
    idx_pos = torch.where(y_true == 1.0)[0]
    idx_neg = torch.where(y_true == 0.0)[0]

    n_pos = idx_pos.size(0)
    n_neg = idx_neg.size(0)

    idx_pos_ex = idx_pos.repeat(n_neg)
    idx_neg_ex = idx_neg.repeat_interleave(n_pos)

    muA = est_means[0:1]
    muP = est_means[idx_pos_ex]
    muN = est_means[idx_neg_ex]

    varA = est_mvars[0:1]
    varP = est_mvars[idx_pos_ex]
    varN = est_mvars[idx_neg_ex]

    probs_ = calculate_lik_prob(muA, muP, muN, varA, varP, varN)
    loss_ = reduce_sum(log(probs_))

    const_ = NmulPnN / (alpha * float(n_pos * n_neg))
    return -const_ * loss_

def calculate_lik_prob(muA, muP, muN, varA, varP, varN, margin=0.0):
    muA2 = square(muA)
    muP2 = square(muP)
    muN2 = square(muN)

    varP2 = square(varP)
    varN2 = square(varN)

    mu = reduce_sum(muP2 + varP - muN2 - varN - 2 * multiply(muA, muP - muN), dim=1)

    T1 = varP2 + 2 * multiply(muP2, varP) + 2 * multiply(varA + muA2, varP + muP2) \
         - 2 * multiply(muA2, muP2) - 4 * multiply(muA, multiply(muP, varP))
    T2 = varN2 + 2 * multiply(muN2, varN) + 2 * multiply(varA + muA2, varN + muN2) \
         - 2 * multiply(muA2, muN2) - 4 * multiply(muA, multiply(muN, varN))
    T3 = 4 * multiply(muP, multiply(muN, varA))
    sigma = torch.sqrt(torch.clamp(reduce_sum(2 * T1 + 2 * T2 - 2 * T3, dim=1), min=0.0))

    dist = Normal(loc=mu, scale=sigma)
    probs_ = dist.cdf(torch.tensor(margin, device=mu.device))
    return probs_

# Main DGP-RF Class
class DGP_RF:
    def __init__(self, data_X, data_Y, trn_index, setting, str_filepath=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Calculate NpNm
        N_pos = np.sum(data_Y[trn_index] == 1)
        self.NpNm = 0.5 * ((N_pos * (N_pos - 1)) * np.sum(data_Y[trn_index] == 0))

        # Settings
        self.max_iter = setting.max_iter
        self.iter_print = setting.iter_print
        self.sub_Ni = setting.sub_Ni
        self.batch_size = setting.batch_size
        self.n_RF = setting.n_RF
        self.regul_const = float(setting.regul_const)
        self.alpha = setting.alpha

        # Feature dimensions
        fea_dims_sub = [100] * setting.n_layers
        fea_dims = np.array([data_X.data_mat[0].shape[1]] + fea_dims_sub)

        # Data
        self.data_X = data_X
        self.trn_index = trn_index
        self.Ytrn = data_Y[trn_index]
        self.Y = np.reshape(data_Y, [-1])
        self.pos_idx = np.intersect1d(np.where(self.Y == 1.0)[0], trn_index)
        self.neg_idx = np.intersect1d(np.where(self.Y == 0.0)[0], trn_index)

        # Model
        self.model = DGP_RF_Embeddings(fea_dims, self.n_RF).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7)

        self.mat_trn_est = None

        # Load or train model
        if str_filepath and path.exists(str_filepath + '.pt'):
            self.model.load_state_dict(torch.load(str_filepath + '.pt'))
            print(f'Loaded trained model: {str_filepath}')
        else:
            self.model_fit()
            if str_filepath:
                torch.save(self.model.state_dict(), str_filepath + '.pt')

    def run_optimization(self, X, X_idx, Y, regul_const=1e-2):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        X_idx = torch.tensor(X_idx, dtype=torch.int32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        est_means, est_vars = self.model(X, X_idx)
        reg_ = self.model.cal_regul()

        # Compute loss
        loss = prob_triplet_loss(Y, est_means, est_vars, self.NpNm, self.alpha)
        obj = loss + regul_const * reg_

        # Backward pass
        obj.backward()
        self.optimizer.step()

        return obj.item()

    def model_fit(self):
        n_pos = max(int(round(self.batch_size / 2)), 1)
        n_neg = n_pos
        iters_Pos = len(self.pos_idx)

        for epoch in range(self.max_iter):
            eta = 1.0 if epoch < 10 else 1.0
            obj = 0.0

            for iter in range(iters_Pos):
                anc_idx = self.pos_idx[iter]
                pos_idx = np.random.choice(
                    np.setdiff1d(self.pos_idx, anc_idx),
                    min(n_pos, len(self.pos_idx) - 1),
                    replace=False
                )
                pos_idx = np.concatenate(([anc_idx], pos_idx))
                neg_idx = np.random.choice(self.neg_idx, min(n_neg, len(self.neg_idx)), replace=False)

                index_vec = np.concatenate((pos_idx, neg_idx))
                set_indices = self.mark_subImgs(self.data_X, index_vec, sub_Ni=self.sub_Ni)

                X, X_idx = self.gen_input_fromList(self.data_X, index_vec, set_indices[0])
                Y = self.Y[index_vec]
                Y[0] = -1  # Anchor node

                obj += self.run_optimization(X, X_idx, Y, eta * self.regul_const)

            print(f'trn Obj: ({epoch}: {obj/iters_Pos:.3f})')

            if self.iter_print and (epoch == self.max_iter - 1):
                out_ests = self.predict(self.trn_index, sub_Ni=self.sub_Ni, rep_num=1, flag_trndata=True)
                auc_val = roc_auc_score(self.Ytrn, out_ests)
                print(f' trnAUC = ({auc_val:.3f})')

    def mark_subImgs(self, data_X, index_vec, sub_Ni, rep_num=1, flag_AllIns=False):
        Nis = np.hstack([data_X.Nis[idx] for idx in index_vec])
        set_indices = []

        for _ in range(rep_num):
            set_indices_sub = []
            for Ni in Nis:
                if not flag_AllIns:
                    idx_selected = np.sort(np.random.choice(
                        np.arange(Ni), size=min(Ni, sub_Ni), replace=False
                    ))
                else:
                    idx_selected = np.arange(Ni)
                set_indices_sub.append(idx_selected)
            set_indices.append(set_indices_sub)

        return set_indices

    def gen_input_fromList(self, data_X, index_vec, set_indices):
        Nis = []
        for mn_i, idx in enumerate(index_vec):
            idx_selected = set_indices[mn_i]
            Xsub = data_X.data_mat[idx][idx_selected]
            Nis.append(len(idx_selected))

            if mn_i == 0:
                X = Xsub
            else:
                X = np.concatenate((X, Xsub), axis=0)

        X_idx = [cnt * np.ones((Ni,), dtype=np.int32) for cnt, Ni in enumerate(Nis)]
        X_idx = np.concatenate(X_idx)

        return X, X_idx

    def select_sub_percnt(self, Xvec, pert=5):
        X_sorted, _ = torch.sort(Xvec, dim=-1)
        N_ = Xvec.size(-1)
        n_rem = int(np.floor(N_ * pert / 100))
        idx_selected = torch.arange(n_rem + 1, N_ - n_rem, device=Xvec.device)
        return X_sorted[..., idx_selected]

    def predict(self, tst_index, data_set_=None, sub_Ni=None, rep_num=1, flag_trndata=False):
        if data_set_ is None:
            data_set_ = self.data_X
        if sub_Ni is None:
            sub_Ni = self.sub_Ni

        self.model.eval()
        with torch.no_grad():
            # Training embeddings
            means_trn, vars_trn = self.model_eval(
                self.trn_index, data_set_=self.data_X, sub_Ni=sub_Ni, rep_num=rep_num
            )

            # Test embeddings
            if flag_trndata:
                means_tst, vars_tst = means_trn, vars_trn
            else:
                means_tst, vars_tst = self.model_eval(
                    tst_index, data_set_=data_set_, sub_Ni=sub_Ni, rep_num=rep_num
                )

            # Compute probabilities
            idx_pos = np.where(self.Ytrn == 1.0)[0]
            idx_neg = np.where(self.Ytrn == 0.0)[0]
            N_pos = len(idx_pos)
            N_neg = len(idx_neg)

            means_trn_pos = means_trn[idx_pos]
            means_trn_neg = means_trn[idx_neg]
            vars_trn_pos = vars_trn[idx_pos]
            vars_trn_neg = vars_trn[idx_neg]

            idx_pos_ex = torch.arange(N_pos).repeat(N_neg)
            idx_neg_ex = torch.arange(N_neg).repeat_interleave(N_pos)

            Y_probs = np.zeros((len(tst_index), 1))
            if flag_trndata:
                for mn_i in range(len(tst_index)):
                    muA = means_tst[mn_i:mn_i+1]
                    varA = vars_tst[mn_i:mn_i+1]

                    if self.Ytrn[mn_i] == 1.0:
                        idx = np.sum(self.Ytrn[:mn_i + 1] == 1)
                        idx_selected = ~(idx_pos_ex == (idx - 1))
                    else:
                        idx = np.sum(self.Ytrn[:mn_i + 1] == 0)
                        idx_selected = ~(idx_neg_ex == (idx - 1))

                    idx_pos_ex_new = idx_pos_ex[idx_selected]
                    idx_neg_ex_new = idx_neg_ex[idx_selected]

                    muP = means_trn_pos[idx_pos_ex_new]
                    muN = means_trn_neg[idx_neg_ex_new]
                    varP = vars_trn_pos[idx_pos_ex_new]
                    varN = vars_trn_neg[idx_neg_ex_new]

                    prob_sub = calculate_lik_prob(muA, muP, muN, varA, varP, varN)
                    Y_probs[mn_i] = torch.mean(prob_sub).cpu().numpy()
            else:
                muP = means_trn_pos[idx_pos_ex]
                muN = means_trn_neg[idx_neg_ex]
                varP = vars_trn_pos[idx_pos_ex]
                varN = vars_trn_neg[idx_neg_ex]

                for mn_i in range(len(tst_index)):
                    muA = means_tst[mn_i:mn_i+1]
                    varA = vars_tst[mn_i:mn_i+1]
                    prob_pos = calculate_lik_prob(muA, muP, muN, varA, varP, varN)
                    Y_probs[mn_i] = torch.mean(prob_pos).cpu().numpy()

        return Y_probs

    def model_eval(self, tst_index, data_set_, sub_Ni, rep_num, batch_size=5):
        Ntst = len(tst_index)
        means_set = []
        vars_set = []

        for mn_i in range(int(np.ceil(Ntst / batch_size))):
            sub_idx = range(mn_i * batch_size, min((mn_i + 1) * batch_size, Ntst))
            index_vec = tst_index[list(sub_idx)]

            set_indices = self.mark_subImgs(data_set_, index_vec, sub_Ni=sub_Ni, rep_num=rep_num)

            for mn_sub in range(rep_num):
                X, X_idx = self.gen_input_fromList(data_set_, index_vec, set_indices[mn_sub])
                X = torch.tensor(X, dtype=torch.float32).to(self.device)
                X_idx = torch.tensor(X_idx, dtype=torch.int32).to(self.device)

                out_means_sub, out_vars_sub = self.model(X, X_idx)
                out_means_sub = out_means_sub.unsqueeze(-1)
                out_vars_sub = out_vars_sub.unsqueeze(-1)

                if mn_sub == 0:
                    out_means, out_vars = out_means_sub, out_vars_sub
                else:
                    out_means = torch.cat((out_means, out_means_sub), dim=2)
                    out_vars = torch.cat((out_vars, out_vars_sub), dim=2)

            if mn_i == 0:
                means_set, vars_set = out_means, out_vars
            else:
                means_set = torch.cat((means_set, out_means), dim=0)
                vars_set = torch.cat((vars_set, out_vars), dim=0)

        return means_set, vars_set