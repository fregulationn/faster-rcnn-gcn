import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from model.utils.config import cfg
import numpy as np


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        #output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class CGCN(torch.nn.Module):
    def __init__(self, n_feat, n_hid, dropout, num_classes, t=0, adj_file = None):
        super(CGCN, self).__init__()

        # self.num_class = num_classes - 1
        self.num_class = num_classes
        self.n_feat =  n_feat
        self.Nms_GCN = GCN(self.n_feat, n_hid, self.n_feat, dropout)

        _adj = self.gen_A(self.num_class , t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        
    def forward(self, roi_socres):
        roi_socres = F.softmax(roi_socres, dim=1)
        # x2 = F.softmax(x.reshape(-1,1),dim = 0).reshape(-1, 20) #Max 0.0013
        
        adj = self.gen_adj(self.A).detach()

        x1, _ = torch.max(roi_socres,0)
        x1 = x1.view(-1,1)

        x1 = self.Nms_GCN(x1, adj)
        # x1 = F.relu(self.gc2(x1, adj))

        if cfg.GCN.REGULAR:
            regular_term = torch.sum(torch.mul(torch.sub(x1,1),torch.sub(x1,1)))
        else:
            regular_term = torch.tensor(0).float().cuda()

        return torch.mul(roi_socres, x1.view(1,-1)), regular_term
    
    def gen_A(self, num_classes, t, adj_file):
        import pickle
        result = pickle.load(open(adj_file, 'rb'))
        _adj = result['adj']
        _nums = result['nums']
        _nums = _nums[:, np.newaxis]
        _adj = _adj / _nums
        _adj[_adj < t] = 0
        _adj[_adj >= t] = 1
        _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
        _adj = _adj + np.identity(num_classes, np.int)
        return _adj

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    # def forward(self, input, target, reg_target, prop_type):
    #     if not self.test_mode:
    #         return self.train_forward(input, target, reg_target, prop_type)
    #     else:
    #         return self.test_forward(input)

    # def get_optim_policies(self):

    #     normal_weight = []
    #     normal_bias = []

    #     for m in self.modules():
    #         if isinstance(m, torch.nn.Linear):
    #             ps = list(m.parameters())
    #             normal_weight.append(ps[0])
    #             if len(ps) == 2:
    #                 normal_bias.append(ps[1])
    #         elif isinstance(m, GraphConvolution):
    #             ps = list(m.parameters())
    #             normal_weight.append(ps[0])
    #             if len(ps) == 2:
    #                 normal_bias.append(ps[1])
    #         elif len(m._modules) == 0:
    #             if len(list(m.parameters())) > 0:
    #                 raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    #     return [
    #         {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
    #          'name': "normal_weight"},
    #         {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
    #          'name': "normal_bias"},
    #     ]
    




    
   