import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import scipy.signal
from gym.spaces import Box, Discrete


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class MLP(nn.Module):
    def __init__(self, in_features,
                 hidden_sizes=(32,),
                 activation=nn.Tanh, 
                 output_activation=None):
        super(MLP, self).__init__()
        
        # first layer
        modules = nn.ModuleList([
            nn.Linear(in_features, out_features=hidden_sizes[0]),
            activation()
            ])
        
        # hidden
        for i, h in enumerate(hidden_sizes[1:-1]):
            modules.append(nn.Linear(
                in_features=hidden_sizes[i],
                out_features=h))
            modules.append(activation())
        
        # last
        modules.append(
            nn.Linear(in_features=hidden_sizes[-2],
                      out_features=hidden_sizes[-1]))
        if output_activation is not None:
            modules.append(output_activation())

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class CategoricalPolicy(nn.Module):
    def __init__(self, in_features,
                 hidden_sizes,
                 activation,
                 action_space):
        super(CategoricalPolicy, self).__init__()
        
        self.act_dim = action_space.n

        self.logits = MLP(in_features,
                          hidden_sizes=list(hidden_sizes)+[self.act_dim],
                          activation=activation)

    def forward(self, x, a=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)

        pi = policy.sample()

        logp_pi = policy.log_prob(pi)
        if a is not None:
            logp = policy.log_prob(a)
        else:
            logp = None

        return pi, logp, logp_pi


class GaussianPolicy(nn.Module):
    def __init__(self, in_features,
                 hidden_sizes,
                 activation,
                 output_activation,
                 action_space):
        super(GaussianPolicy, self).__init__()

        self.act_dim = action_space.shape[0]
        self.mu = MLP(in_features,
                      hidden_sizes=list(hidden_sizes)+[self.act_dim],
                      activation=activation,
                      output_activation=output_activation)
        
        self.log_std = nn.Parameter(-0.5*torch.ones(self.act_dim, dtype=torch.float32))
        
    def forward(self, x, a=None):
        mu = self.mu(x)
        std = torch.exp(self.log_std)
        policy = Normal(mu, std)

        pi = policy.sample()
        logp_pi = torch.sum(policy.log_prob(pi), dim=1)
        
        if a is not None:
            logp = torch.sum(policy.log_prob(a), dim=1)
        else:
            logp = None
        
        return pi, logp, logp_pi


class ActorCritic(nn.Module):
    def __init__(self, in_features,
                 hidden_sizes=(64, 64),
                 activation=nn.Tanh,
                 output_activation=None,
                 policy=None,
                 action_space=None):
        super(ActorCritic, self).__init__()

        if policy is None and isinstance(action_space, Box):
            self.policy = GaussianPolicy(in_features,
                                         hidden_sizes,
                                         activation,
                                         output_activation,
                                         action_space)
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(in_features,
                                            hidden_sizes,
                                            activation,
                                            action_space)
        else:
            self.policy = policy
        
        self.value_function = MLP(in_features, 
                                  list(hidden_sizes)+[1],
                                  activation)
    
    def forward(self, x, a=None):
        pi, logp, logp_pi = self.policy(x, a)

        v = self.value_function(x)

        return pi, logp, logp_pi, v
