import torch
from spinup.utils.mpi_tools import broadcast, mpi_avg


def sync_all_params(model):
    for param in model.parameters():
        data = param.data.numpy()
        broadcast(data)
        param.data = torch.Tensor(data)

def average_gradients(model):
    for param in model.parameters():
        param.grad.data = torch.Tensor(mpi_avg(param.grad.data.numpy()))
