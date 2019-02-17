#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

### YOUR CODE HERE for part 1h

class Highway(nn.Module):
  def __init__(self, embed_size):
    super(Highway, self).__init__()
    self.proj = nn.Linear(embed_size, embed_size)
    self.gate = nn.Linear(embed_size, embed_size)

  def forward(self, x):
    x_proj = torch.relu(self.proj(x))
    x_gate = torch.sigmoid(self.gate(x))
    x_highway = x_gate * x_proj + (1-x_gate)*x
    return x_highway

### END YOUR CODE
