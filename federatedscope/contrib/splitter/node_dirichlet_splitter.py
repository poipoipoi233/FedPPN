import torch
from federatedscope.register import register_splitter
import numpy as np
from torch_geometric.utils.subgraph import subgraph
from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice
from torch_geometric.data import Data
import copy
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, from_networkx

EPSILON = 1e-5


class NodeDirichletSplitter(BaseSplitter):
    def __init__(self,
                 client_num,
                 alpha=0.5):
        super(NodeDirichletSplitter, self).__init__(client_num)

        self.alpha = alpha

    def __call__(self, data, **kwargs):
        data.index_orig = torch.arange(data.num_nodes)
        # G = to_networkx(
        #     data,
        #     node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
        #     to_undirected=True)
        # nx.set_node_attributes(G,
        #                        dict([(nid, nid)
        #                              for nid in range(nx.number_of_nodes(G))]),
        #                        name="index_orig")

        client_node_idx = {idx: [] for idx in range(self.client_num)}

        indices = np.random.permutation(data.num_nodes)
        label = data.y
        idx_slice = dirichlet_distribution_noniid_slice(label,
                                                        self.client_num,
                                                        self.alpha,
                                                        prior=None)

        for idx in range(self.client_num):
            client_node_idx[idx] = indices[idx_slice[idx]]

        graphs = []
        for owner in client_node_idx:
            nodes = torch.from_numpy(client_node_idx[owner]).long()
            sub_edge_index, _ = subgraph(nodes, data.edge_index, relabel_nodes=True)
            sub_g = copy.deepcopy(Data(
                x=data.x[nodes],
                edge_index=sub_edge_index,
                y=data.y[nodes],
                train_mask=data.train_mask[nodes],
                val_mask=data.val_mask[nodes],
                test_mask=data.test_mask[nodes],
                index_orig=nodes
            ))
            graphs.append(sub_g)

        return graphs


def call_my_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'node_dirichlet_splitter':
        splitter = NodeDirichletSplitter(client_num, **kwargs)
        return splitter


register_splitter('node_dirichlet_splitter', call_my_splitter)
