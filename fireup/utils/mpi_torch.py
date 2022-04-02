import torch

from fireup.utils.mpi_tools import broadcast, mpi_avg, num_procs


def setup_pytorch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)


def sync_all_params(param, root=0):
    data = torch.nn.utils.parameters_to_vector(param).detach().numpy()
    broadcast(data, root)
    torch.nn.utils.vector_to_parameters(torch.from_numpy(data), param)


def average_gradients(param_groups):
    for param_group in param_groups:
        for p in param_group["params"]:
            if p.requires_grad:
                p.grad.data.copy_(torch.Tensor(mpi_avg(p.grad.data.numpy())))
