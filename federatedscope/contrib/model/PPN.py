import torch.nn.functional as F
from federatedscope.register import register_model
from federatedscope.contrib.model.label_prop import LabelPropagation, learnable_LabelPropagation
import torch
import torch.nn as nn
from torch.nn import Parameter

"""
Prototype Propagation Network (PPN)
"""


class PPN(nn.Module):
    def __init__(self, num_layers: int, alpha: float):
        super(PPN, self).__init__()
        self.LP = LabelPropagation(num_layers, alpha)

    def forward(self, train_mask, global_protos, labels_all, edge_index):
        # train_mask: (N_all, )
        # global_protos: (N_class, feature_dim)
        proto_reps = self.initialize_prototype_reps(train_mask, global_protos, labels_all)
        labels = self.LP(y=proto_reps, edge_index=edge_index, train_mask=None)

        not_updated_indices = (labels == 0).all(dim=-1).nonzero().squeeze()

        num_not_updated = len(not_updated_indices)
        return labels, num_not_updated, not_updated_indices

    def initialize_prototype_reps(self, train_mask, global_protos, labels_all):
        size = (train_mask.shape[0], global_protos.shape[1])
        proto_rep = torch.zeros(size, device=global_protos.device)  # (N_all, feature_dim)
        proto_rep[train_mask] = global_protos[labels_all[train_mask]]  # (N_all, feature_dim)
        return proto_rep


class Learnable_PPN(nn.Module):
    def __init__(self, num_layers: int):
        super(Learnable_PPN, self).__init__()
        self.LP = learnable_LabelPropagation(num_layers)
        self.alpha = Parameter(torch.tensor(0.5))

    def forward(self, train_mask, global_protos, labels_all, edge_index):
        # train_mask: (N_all, )
        # global_protos: (N_class, feature_dim)
        proto_reps = self.initialize_prototype_reps(train_mask, global_protos, labels_all)
        labels = self.LP(y=proto_reps, edge_index=edge_index, alpha=self.alpha)
        return labels

    def initialize_prototype_reps(self, train_mask, global_protos, labels_all):
        size = (train_mask.shape[0], global_protos.shape[1])
        proto_rep = torch.zeros(size, device=global_protos.device)  # (N_all, feature_dim)
        proto_rep[train_mask] = global_protos[labels_all[train_mask]]  # (N_all, feature_dim)
        return proto_rep
