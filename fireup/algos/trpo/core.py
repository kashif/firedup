import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from gym.spaces import Box, Discrete


def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))


def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]


def flat_grad(f, param, **kwargs):
    return parameters_to_vector(torch.autograd.grad(f, param, **kwargs))


def hessian_vector_product(f, policy, x):
    # for H = grad**2 f, compute Hx
    g = flat_grad(f, policy.parameters(), create_graph=True)
    return flat_grad((g * x.detach()).sum(), policy.parameters(), retain_graph=True)


class MLP(nn.Module):
    def __init__(
        self,
        layers,
        activation=torch.tanh,
        output_activation=None,
        output_squeeze=False,
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x


class CategoricalPolicy(nn.Module):
    def __init__(
        self, in_features, hidden_sizes, activation, output_activation, action_dim
    ):
        super(CategoricalPolicy, self).__init__()

        self.logits = MLP(
            layers=[in_features] + list(hidden_sizes) + [action_dim],
            activation=activation,
        )

    def forward(self, x, a=None, old_logits=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)

        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()

        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None

        if old_logits is not None:
            old_policy = Categorical(logits=old_logits)
            d_kl = kl_divergence(old_policy, policy).mean()
        else:
            d_kl = None

        info = {"old_logits": logits.detach().numpy()}

        return pi, logp, logp_pi, info, d_kl


class GaussianPolicy(nn.Module):
    def __init__(
        self, in_features, hidden_sizes, activation, output_activation, action_dim
    ):
        super(GaussianPolicy, self).__init__()

        self.mu = MLP(
            layers=[in_features] + list(hidden_sizes) + [action_dim],
            activation=activation,
            output_activation=output_activation,
        )
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim, dtype=torch.float32))

    def forward(self, x, a=None, old_log_std=None, old_mu=None):
        mu = self.mu(x)
        policy = Normal(mu, self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None

        if (old_mu is not None) or (old_log_std is not None):
            old_policy = Normal(old_mu, old_log_std.exp())
            d_kl = kl_divergence(old_policy, policy).mean()
        else:
            d_kl = None

        info = {
            "old_mu": np.squeeze(mu.detach().numpy()),
            "old_log_std": self.log_std.detach().numpy(),
        }

        return pi, logp, logp_pi, info, d_kl


class ActorCritic(nn.Module):
    def __init__(
        self,
        in_features,
        action_space,
        hidden_sizes=(64, 64),
        activation=torch.tanh,
        output_activation=None,
        policy=None,
    ):
        super(ActorCritic, self).__init__()

        if policy is None and isinstance(action_space, Box):
            self.policy = GaussianPolicy(
                in_features,
                hidden_sizes,
                activation,
                output_activation,
                action_dim=action_space.shape[0],
            )
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(
                in_features,
                hidden_sizes,
                activation,
                output_activation,
                action_dim=action_space.n,
            )
        else:
            self.policy = policy(
                in_features, hidden_sizes, activation, output_activation, action_space
            )

        self.value_function = MLP(
            layers=[in_features] + list(hidden_sizes) + [1],
            activation=activation,
            output_squeeze=True,
        )

    def forward(self, x, a=None, **kwargs):
        pi, logp, logp_pi, info, d_kl = self.policy(x, a, **kwargs)
        v = self.value_function(x)

        return pi, logp, logp_pi, info, d_kl, v
