import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete


LOG_STD_MAX = 2
LOG_STD_MIN = -20

def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class MLP(nn.Module):
    def __init__(self, layers, activation=torch.tanh, output_activation=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            return self.layers[-1](x)
        else:
            return self.output_activation(self.layers[-1](x))


class GaussianPolicy(nn.Module):
    def __init__(self, in_features,
                 hidden_sizes,
                 activation,
                 output_activation,
                 action_space):
        super(GaussianPolicy, self).__init__()

        action_dim = action_space.shape[0]
        self.action_scale = action_space.high[0]

        self.net = MLP(layers=[in_features]+list(hidden_sizes),
                      activation=activation,
                      output_activation=activation)

        modules = [nn.Linear(in_features=list(hidden_sizes)[-1],
                             out_features=action_dim)]
        if output_activation is not None:
            modules.append(output_activation())
        self.mu = nn.Sequential(*modules)

        """
        Because this algorithm maximizes trade-off of reward and entropy,
        entropy must be unique to state---and therefore log_stds need
        to be a neural network output instead of a shared-across-states
        learnable parameter vector. But for deep Relu and other nets,
        simply sticking an activationless dense layer at the end would
        be quite bad---at the beginning of training, a randomly initialized
        net could produce extremely large values for the log_stds, which
        would result in some actions being either entirely deterministic
        or too random to come back to earth. Either of these introduces
        numerical instability which could break the algorithm. To
        protect against that, we'll constrain the output range of the
        log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is
        slightly different from the trick used by the original authors of
        SAC---they used torch.clamp instead of squashing and rescaling.
        I prefer this approach because it allows gradient propagation
        through log_std where clipping wouldn't, but I don't know if
        it makes much of a difference.
        """
        self.log_std = nn.Sequential(nn.Linear(in_features=list(hidden_sizes)[-1],
                                                out_features=action_dim),
                                     nn.Tanh())

    def forward(self, x):
        output = self.net(x)
        mu = self.mu(output)
        log_std = self.log_std(output)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        policy = Normal(mu, scale=torch.exp(log_std))
        pi = policy.sample()
        logp_pi = torch.sum(policy.log_prob(pi), dim=1)

        mu, pi, logp_pi = self._apply_squashing_func(mu, pi, logp_pi)

        # make sure actions are in correct range
        mu *= self.action_scale
        pi *= self.action_scale

        return mu, pi, logp_pi

    def _clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        return x + ((u - x)*clip_up + (l - x)*clip_low).detach()

    def _apply_squashing_func(self, mu, pi, logp_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)

        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        logp_pi -= torch.sum(
            torch.log(self._clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), dim=1)

        return mu, pi, logp_pi

class ActorCritic(nn.Module):
    def __init__(self, in_features, action_space,
                 hidden_sizes=(400, 300),
                 activation=torch.relu,
                 output_activation=None,
                 policy=GaussianPolicy):
        super(ActorCritic, self).__init__()

        action_dim = action_space.shape[0]

        self.policy = policy(in_features,
                             hidden_sizes,
                             activation,
                             output_activation,
                             action_space)

        self.vf_mlp = MLP([in_features]+list(hidden_sizes)+[1],
                          activation)

        self.q1 = MLP([in_features+action_dim]+list(hidden_sizes)+[1],
                      activation)

        self.q2 = MLP([in_features+action_dim]+list(hidden_sizes)+[1],
                      activation)

    def forward(self, x, a):
        mu, pi, logp_pi = self.policy(x)

        q1 = self.q1(torch.cat((x, a), dim=-1))
        q1_pi = self.q1(torch.cat((x, pi), dim=-1))

        q2 = self.q2(torch.cat((x, a), dim=-1))
        q2_pi = self.q2(torch.cat((x, pi), dim=-1))

        v = self.vf_mlp(x)

        return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v
