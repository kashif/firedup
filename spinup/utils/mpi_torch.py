import torch
from spinup.utils.mpi_tools import broadcast, mpi_avg


def sync_all_params(params, root=0):
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]

    for _, p in params:
        data = p.data.numpy()
        broadcast(data, root)
        p.data.copy_(torch.Tensor(data))

def average_gradients(param_groups):
    for param_group in param_groups:
        for p in param_group['params']:
            if p.requires_grad:
                p.grad.data.copy_(torch.Tensor(mpi_avg(p.grad.data.numpy())))
