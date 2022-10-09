import torch
import torch.nn as nn
from model import NodeAttention, SemanticAttention
import torch.nn.functional as F

class HAN(nn.Module):

    def __init__(self, input_features, n_hid, head_number, meta_path_number, dropout=0.5, alpha=0.5, aloha = 0.5, link_prediction_layer = 0):
        super(HAN, self).__init__()

        self.node_attention = NodeAttention(input_features, n_hid, head_number, dropout, alpha, meta_path_number)

        self.semantic_attention = SemanticAttention(n_hid, n_hid, aloha, link_prediction_layer)

        self.dropout = dropout

        self.distmult = Dis(n_hid)

    def forward(self, features, adjs):

        Z = self.node_attention(features, adjs)

        Z = F.dropout(Z, self.dropout, training=self.training)

        Z = self.semantic_attention(Z)

        Z = F.dropout(Z, self.dropout, training=self.training)

        score = self.distmult(Z)

        return score

class Dis(nn.Module):
    def __init__(self,f_dim,w_init='standard-normal',):
        super(Dis, self).__init__()
        self.w_init = w_init
        self.relations = nn.Parameter(torch.FloatTensor(f_dim))
        self.initialise_parameters()

    def initialise_parameters(self):
        # Weights
        init = torch.nn.init.normal_
        init(self.relations)

    def compute_penalty(self):
        return self.relations.pow(2).sum()

    def forward(self, fea):
        r = self.relations.repeat(fea.shape[0],1)
        scores = (fea * r).matmul(fea.T)
        penalty = self.compute_penalty()
        return scores , penalty


