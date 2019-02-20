#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

### YOUR CODE HERE for part 1h

class Highway(nn.Module):
    """ An implementation of Highway network."""

    def __init__(self, dim):
        """Initialize Highway module.

        @param dim (int): The number of features (size of last dimension in 
            input tensor).
        """
        super(Highway, self).__init__()
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        """Passes 'x' through the highway network.

        @param x (tensor.Tensor): Input tensor such that the size of the last
            dimension is 'dim' (specificed in constructor).
        
        @returns tensor.Tensor : Output tensor with the same shape as input.
        """
        x_proj = torch.relu(self.proj(x))
        x_gate = torch.sigmoid(self.gate(x))
        x_highway = x_gate * x_proj + (1-x_gate) * x
        return x_highway

### END YOUR CODE
