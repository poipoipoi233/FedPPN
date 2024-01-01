import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import (
    index_to_mask,
    to_undirected,
)


def set_dataset_attr(dataset, name, value, size):
    dataset._data_list = None
    dataset.data[name] = value
    if dataset.slices is not None:
        dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)


def load_obgn_arxiv(path):
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     root=path
                                     )

    splits = dataset.get_idx_split()
    split_names = ['train_mask', 'val_mask', 'test_mask']
    for i, key in enumerate(splits.keys()):
        mask = index_to_mask(splits[key], size=dataset.data.y.shape[0])
        set_dataset_attr(dataset, split_names[i], mask, len(mask))
    edge_index = to_undirected(dataset.data.edge_index)
    set_dataset_attr(dataset, 'edge_index', edge_index,
                     edge_index.shape[1])

    data = dataset[0]
    data.y = data.y.squeeze()
    return data
