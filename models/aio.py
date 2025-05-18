# Refactored version of the original DGP-RF model implementation using TensorFlow
# This refactor improves readability, modularization, and maintainability

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, initializers
from tensorflow.keras.layers import Layer
from sklearn.metrics import roc_auc_score
from os import path

# --- Utility functions ---
from models.tf_commands import *  # assuming these include mat_mul, square, exp, reduce_sum, etc.
dist_Normal = tf.compat.v1.distributions.Normal

# --- Variational Bayesian Layer ---
class VBLayer(Layer):
    def __init__(self, units, dim_input, is_ReLUoutput=False, use_bias=True,
                 kernel_initializer='glorot_normal', bias_initializer='zeros', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.d_input = dim_input
        self.is_ReLUoutput = is_ReLUoutput
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.prior_prec = 1.0

    def build(self, input_shape):
        self.w_mus = self.add_weight(shape=(self.d_input, self.units), initializer=self.kernel_initializer, name='kernel_mu')
        self.w_logsig2 = self.add_weight(shape=(self.d_input, self.units), initializer=self.kernel_initializer, name='kernel_logsig2')
        self.b = self.add_weight(shape=(1, self.units), initializer=self.bias_initializer, name='bias')
        if self.is_ReLUoutput:
            self.gamma = self.add_weight(shape=(1, self.units), initializer=self.bias_initializer, name='omega_ard')

    def call(self, in_means, in_vars):
        out_means = mat_mul(in_means, self.w_mus) + self.b

        if not self.is_ReLUoutput:
            m2s2_w = square(self.w_mus) + exp(self.w_logsig2)
            out_vars = mat_mul(in_vars, m2s2_w) + mat_mul(square(in_means), exp(self.w_logsig2))
        else:
            pZ = sigmoid(out_means)
            factor_ = exp(0.5 * self.gamma) * (2 / np.sqrt(self.units))
            out_means = multiply(multiply(pZ, out_means), factor_)
            m2s2_w = square(self.w_mus) + exp(self.w_logsig2)
            term1 = mat_mul(in_vars, m2s2_w) if in_vars is not None else 0
            term2 = mat_mul(square(in_means), exp(self.w_logsig2))
            term3 = square(mat_mul(in_means, self.w_mus) + self.b)
            out_vars = multiply(pZ, term1 + term2) + multiply(pZ * (1 - pZ), term3)
            out_vars = multiply(square(factor_), out_vars)

        return out_means, out_vars

    def calculate_kl(self):
        w_logsig2 = tf.clip_by_value(self.w_logsig2, -11, 11)
        kl = 0.5 * reduce_sum(self.prior_prec * (square(self.w_mus) + exp(w_logsig2)) - w_logsig2 - log(self.prior_prec))
        return kl

# --- Embedding model ---
class DGP_RF_Embeddings(Model):
    def __init__(self, fea_dims, num_RF):
        super().__init__()
        self.layers_list = []
        for i in range(len(fea_dims) - 1):
            self.layers_list.append(VBLayer(num_RF, fea_dims[i], is_ReLUoutput=True))
            self.layers_list.append(VBLayer(fea_dims[i + 1], num_RF))

    def call(self, X, X_idx):
        inter_means = X
        inter_vars = tf.zeros_like(X)

        for layer in self.layers_list:
            inter_means, inter_vars = layer(inter_means, inter_vars)

        out_means, out_vars = inter_means, inter_vars
        mat_tmp1 = divide(1, out_vars)
        embedd_vars = divide(1, segment_sum(mat_tmp1, segment_ids=X_idx))
        embedd_means = segment_sum(multiply(mat_tmp1, out_means), segment_ids=X_idx)

        return multiply(embedd_vars, embedd_means), embedd_vars

    def cal_regul(self):
        return tf.add_n([layer.calculate_kl() for layer in self.layers_list if hasattr(layer, "calculate_kl")])

# --- Loss Function ---
def Prob_Triplet_loss(y_true, est_means, est_mvars, NmulPnN, alpha=0.5):
    idx_pos = vec_rowwise(tf.where(y_true == 1.0))
    idx_neg = vec_colwise(tf.where(y_true == 0.0))
    n_pos = tf.size(idx_pos)
    n_neg = tf.size(idx_neg)
    idx_pos_ex = vec_flat(K.repeat(idx_pos, n=n_neg))
    idx_neg_ex = vec_flat(K.repeat(idx_neg, n=n_pos))
    muA = tf.gather(est_means, [0], axis=0)
    muP = tf.gather(est_means, idx_pos_ex, axis=0)
    muN = tf.gather(est_means, idx_neg_ex, axis=0)
    varA = tf.gather(est_mvars, [0], axis=0)
    varP = tf.gather(est_mvars, idx_pos_ex, axis=0)
    varN = tf.gather(est_mvars, idx_neg_ex, axis=0)
    probs_ = calculate_lik_prob(muA, muP, muN, varA, varP, varN)
    loss_ = reduce_sum(log(probs_))
    const_ = NmulPnN / (alpha * tf.cast((n_pos * n_neg), dtype=np.float32))
    return -const_ * loss_

# --- Likelihood Probability ---
def calculate_lik_prob(muA, muP, muN, varA, varP, varN, margin=0.0):
    mu = reduce_sum(muP**2 + varP - muN**2 - varN - 2 * muA * (muP - muN), axis=1)
    T1 = varP**2 + 2 * muP**2 * varP + 2 * (varA + muA**2) * (varP + muP**2) - 2 * muA**2 * muP**2 - 4 * muA * muP * varP
    T2 = varN**2 + 2 * muN**2 * varN + 2 * (varA + muA**2) * (varN + muN**2) - 2 * muA**2 * muN**2 - 4 * muA * muN * varN
    T3 = 4 * muP * muN * varA
    sigma = sqrt(tf.maximum(0.0, reduce_sum(2 * T1 + 2 * T2 - 2 * T3, axis=1)))
    return dist_Normal(loc=mu, scale=sigma).cdf(margin)


