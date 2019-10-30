# Welcome to Fired Up in Deep RL!


This is a clone of OpenAI's [Spinning Up](https://github.com/openai/spinningup) in PyTorch. Spinning Up is an awesome educational resource produced by Josh Achiam, a  research scientist at [OpenAI](https://openai.com/), that makes it easier to learn about deep reinforcement learning (deep RL).

## Installation

Fired Up requires Python3, PyTorch, OpenAI Gym, and OpenMPI.

Fired Up is currently only supported on Linux and OSX. It may be possible to install on Windows, though I  haven't tested this OS.

### Installing Python

We recommend installing Python through [Anaconda](https://www.anaconda.com/distribution/#download-section). Anaconda is a Python distribution that includes many useful packages especially for scientific computing, as well as an environment manager called `conda` that makes package management simple.

Download and install Anaconda 2018.x (at time of writing, 2018.12) Python 3.7. Then create a `conda` environment for organizing packages used in Fired Up:

```
conda create -n firedup python=3.7
```

To use Python from the environment you just created, activate the environment with:

```
source activate firedup
```

You can alternatively use [virtualenv](https://virtualenv.pypa.io/en/latest/) with the Python3 version you have. Just install it via `pip3` and then:

```
virtualenv firedup
```

To activate this virtual environment you need to:

```
source /path/to/firedup/bin/activate
```

### Installing OpenMPI

#### Ubuntu

```
sudo apt update && sudo apt install libopenmpi-dev
```

#### Mac OS X

Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:

```
brew install openmpi
```

### Installing Fired Up

```
git clone https://github.com/kashif/firedup.git
cd firedup
pip install -e .
```

Fired Up defaults to installing everything in Gym **except** the MuJoCo environments.

### Check Your Install

To see if you've successfully installed Fired Up, try running PPO in the `LunarLander-v2` environment with:

```
python -m fireup.run ppo --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999
```

After it finishes training, watch a video of the trained policy with:

```
python -m fireup.run test_policy data/installtest/installtest_s0
```

And plot the results with:

```
python -m fireup.run plot data/installtest/installtest_s0
```

## Algorithms

The following algorithms are implemented in the Fired Up package:

* Vanilla Policy Gradient (VPG)
* Trust Region Policy Optimization (TRPO)
* Proximal Policy Optimization (PPO)
* Deep Q-Network (DQN)
* Deep Deterministic Policy Gradient (DDPG)
* Twin Delayed DDPG (TD3)
* Soft Actor-Critic (SAC)

They are all implemented with MLP (non-recurrent) actor-critics, making them suitable for fully-observed, non-image-based RL environments, e.g. the Gym Mujoco environments.

## Citation

If you use Fired Up in your research please  use the following BibTeX entry:

```BibTeX
@misc{rasulfiredup,
  author =       {Kashif Rasul and Josh Achiam},
  title =        {Fired Up},
  howpublished = {\url{https://github.com/kashif/firedup/}},
  year =         {2019}
}
```
