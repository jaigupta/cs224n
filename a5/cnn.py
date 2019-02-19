#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1i

class CNN(nn.Module):
  def __init__(self, embed_size, kernel_size=5):
    super(CNN, self).__init__()
    self.conv = nn.Conv1d(embed_size, embed_size, kernel_size)
    self.relu = nn.ReLU()

  def forward(self, x):
    sents_len, batch_size, word_len, embed_size = x.shape
    x = x.permute(0, 1, 3, 2)
    x = x.view(-1, embed_size, word_len)
    x = self.conv(x)
    x = self.relu(x)
    x = nn.MaxPool1d(word_len)(x)
    return x.view(sents_len, batch_size, embed_size)

### END YOUR CODE
