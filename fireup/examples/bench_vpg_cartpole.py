import os

import torch
from fireup import vpg
from fireup.utils.run_utils import ExperimentGrid

os.environ["WANDB_MODE"] = "offline"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--num_runs", type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name="vpg-bench")
    eg.add("env_name", "CartPole-v0", "", True)
    eg.add("seed", [10 * i for i in range(args.num_runs)])
    eg.add("epochs", 10)
    eg.add("steps_per_epoch", 4000)
    eg.add("ac_kwargs:hidden_sizes", [(32,), (64, 64)], "hid")
    eg.add("ac_kwargs:activation", [torch.tanh, torch.relu], "")
    eg.run(vpg, num_cpu=args.cpu)
