import torch
from torch import nn

def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss(prediction_model, x, h, node_mask, edge_mask, context):
    bs, n_nodes, n_dims = x.size()
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    assert_correctly_masked(x, node_mask)
    # Here x is a position tensor, and h is a dictionary with keys
    # 'categorical' and 'integer'.
    loss = prediction_model(x, h, node_mask, edge_mask,context)
    return loss
