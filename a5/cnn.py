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
    """ CNN class wraps a convolutional-nn to convert input sequence of 
    character level embeddings for variable length words (but padded to same
    length)into a single word-level embedding of same number of dimensions.
    """

    def __init__(self, embed_size, kernel_size=5):
        """ Init CNN module.

        @param embed_size (int): Size of embedding of each character.
        @param kernel_size (int): Size of kernel for the CNN used for mapping the
            input embedding to a new set of features. The number of features
            produced is the same as input 'embed_size'.
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(embed_size, embed_size, kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Convert a sequence of character level embeeddings of size
        'embed_size' for variable length words (but padded to same length)
        to a single entry with same dimension representing the word-level embedding.
        
        @param x (torch.Tensor): Input tensor of shape (sents_len, batch_size,
            word_len, embed_size) representing the charcter level embeddings.

        @returns (tensor.Torch): A tensor of shape (sents_len, batch_size,
            embed_size) where each entry (of size embed_size) represents the
            word-level embedding.
        """
        # assert x.shape[1] == self.dim
        sents_len, batch_size, word_len, embed_size = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.view(-1, embed_size, word_len)
        x = self.conv(x)
        x = self.relu(x)
        x = torch.max(x, dim=-1)[0]
        return x.view(sents_len, batch_size, embed_size)

### END YOUR CODE
