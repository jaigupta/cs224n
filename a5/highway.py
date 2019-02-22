#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn

class Highway(nn.Module):
    """ An implementation of Highway network."""

    def __init__(self, dim):
        """Initialize Highway module.

        @param dim (int): The number of features (size of last dimension in
            input tensor).
        """
        super(Highway, self).__init__()
        # assert dim > 0, 'Expected positve dim, found {}'.format(dim)
        # self.dim = dim
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        # assert(self.proj.weight.shape == (dim, dim))
        # assert(self.proj.bias.shape == (dim,))
        # assert(self.gate.weight.shape == (dim, dim))
        # assert(self.gate.bias.shape == (dim,))

    def forward(self, x):
        """Passes 'x' through the highway network.

        @param x (tensor.Tensor): Input tensor such that the size of the last
            dimension is 'dim' (specificed in constructor).

        @returns tensor.Tensor : Output tensor with the same shape as input.
        """
        # assert(x.shape[-1] == self.dim)
        x_proj = torch.relu(self.proj(x))
        # assert(x_proj.shape == x.shape)
        x_gate = torch.sigmoid(self.gate(x))
        # assert(x_proj.shape == x.shape)
        x_highway = x_gate * x_proj + (1-x_gate) * x
        # assert(x_highway.shape == x.shape)
        return x_highway

### END YOUR CODE
