import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete


def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class MLP(nn.Module):
    def __init__(self, layers, activation=torch.tanh, output_activation=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation

        gain = nn.init.calculate_gain(activation.__name__)
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)
            nn.init.xavier_uniform_(self.layers[i].weight, gain)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            return self.layers[-1](x)
        else:
            return self.output_activation(self.layers[-1](x))


class ActorCritic(nn.Module):
    def __init__(self, in_features, action_space,
                 hidden_sizes=(400, 300),
                 activation=torch.relu,
                 output_activation=torch.tanh):
        super(ActorCritic, self).__init__()

        action_dim = action_space.shape[0]
        self.action_scale = action_space.high[0]

        self.policy = MLP(layers=[in_features]+list(hidden_sizes)+[action_dim],
                          activation=activation, output_activation=output_activation)
        self.q1 = MLP(layers=[in_features+action_dim]+list(hidden_sizes)+[1],
                      activation=activation)
        self.q2 = MLP(layers=[in_features+action_dim]+list(hidden_sizes)+[1],
                      activation=activation)

    def forward(self, x, a):
        pi = self.action_scale * self.policy(x)

        q1 = self.q1(torch.cat((x, a), dim=1))
        q2 = self.q2(torch.cat((x, a), dim=1))

        q1_pi = self.q1(torch.cat((x, pi), dim=1))

        return pi, q1, q2, q1_pi