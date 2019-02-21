#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
sanity_check.py: sanity checks for assignment 5
Usage:
    sanity_check.py 1h
    sanity_check.py 1i
"""
import json
import math
import pickle
import sys
import time

import numpy as np

from docopt import docopt
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import pad_sents_char, read_corpus, batch_iter
from vocab import Vocab, VocabEntry

from char_decoder import CharDecoder
from cnn import CNN
from highway import Highway
from nmt_model import NMT


import torch
import torch.nn as nn
import torch.nn.utils

#----------
# CONSTANTS
#----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0


def verify_linear_shape(name, model, shape):
    assert type(model) == nn.Linear, 'Expected nn.Linear module for test.'
    assert model.weight.shape == shape, \
            '{} weight matrix: Expected shape: {}, found: {}'.format(
                    name, shape, model.weight.shape)
    expected_bias_shape = (shape[-1],)
    assert model.bias.shape == expected_bias_shape, \
            '{} bias matrix: Expected shape: {}, found: {}'.format(
                    name, expected_bias_shape, model.weight.shape)


def init_weights(model):
    if type(model) == nn.Linear:
        model.weight.data.fill_(0.3)
        if model.bias is not None:
            model.bias.data.fill_(0.1)


def question_1h_sanity_check():
    """ Sanity check for highway.Highway module.
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1h: highway.Highway module")
    print ("-"*80)

    dim = 3
    highway = Highway(dim)

    # Some of these tests are depending on private variables defined within
    # the Highway class. Usually it is not a good idea since we should code
    # on the public interface. But good enough for this case.
    verify_linear_shape("Highway.proj", highway.proj, (3, 3))
    verify_linear_shape("Highway.gate", highway.gate, (3, 3))

    highway.apply(init_weights)

    inp = torch.tensor(
        [[[0.5, 0.9, 0.5],
          [0.8, 12, 1.333333],
          [0.111, 0.2, 0.12],
          [0.6, -0.5, 0.5]],
         [[0.1, -0.4, 0.6],
          [0.4, 0.0, 0.7],
          [0.5, -0.8, 0.75],
          [0.5, 0.6, 0.6]],
         [[0.3, 0.9, 0.09],
          [0.9, 0.4, 0.8],
          [0.6, 0.3, 0.0],
          [0.2, 0.4, 0.6]]])

    expected_output = torch.tensor(
        [[[ 0.6125,  0.7479,  0.6125 ],
          [ 4.2944,  4.4386,  4.3013 ],
          [ 0.1769,  0.2163,  0.1809 ],
          [ 0.4177, -0.0558,  0.3747 ]],
         [[ 0.1493, -0.0771,  0.3756 ],
          [ 0.4182,  0.2605,  0.5364 ],
          [ 0.3520, -0.2220,  0.4624 ],
          [ 0.5713,  0.6065,  0.6065 ]],
         [[ 0.4158,  0.6442,  0.3359 ],
          [ 0.7853,  0.6227,  0.7528 ],
          [ 0.4640,  0.3414,  0.2188 ],
          [ 0.3594,  0.4368,  0.5142 ]]])

    with torch.no_grad():
        output_mat = highway(inp)
    assert(np.allclose(expected_output.numpy(), output_mat.numpy(), rtol=01e-3)), \
            'Expected output to be {}, found {}'.format(expected_output, output_mat)

    print("All Sanity Checks Passed for Question 1h: highway.Highway!")
    print ("-"*80)


def question_1i_sanity_check():
    """ Sanity check for highway.Highway module.
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1h: highway.Highway module")
    print ("-"*80)

    dim = 3
    kernel_size = 4
    cnn = CNN(dim, kernel_size=kernel_size)

    conv_1d = next(x for x in cnn.children())
    assert conv_1d, "nn.Conv1d not found registered with the CNN module."

    cnn.apply(init_weights)

    def verify_sol(inp, expected_output):
        with torch.no_grad():
            output_mat = cnn(inp)
        assert output_mat.shape == (inp.shape[0], inp.shape[1], inp.shape[3])
        assert(np.allclose(expected_output.numpy(), output_mat.numpy(), rtol=01e-3)), \
                'Expected output to be {}, found {}'.format(expected_output, output_mat)

    inp = torch.tensor(
        [
         [[[0.5, 0.9, 0.5],
          [0.8, 12, 1.333333],
          [0.111, 0.2, 0.12],
          [0.111, 0.2, 0.12],
          [0.6, -0.5, 0.5]],
         [[0.1, -0.4, 0.6],
          [0.4, 0.0, 0.7],
          [0.111, 0.2, 0.12],
          [0.5, -0.8, 0.75],
          [0.5, 0.6, 0.6]],
         [[0.3, 0.9, 0.09],
          [0.9, 0.4, 0.8],
          [0.111, 0.2, 0.12],
          [0.6, 0.3, 0.0],
          [0.2, 0.4, 0.6]]],
         [[[1.5, 1.9, 1.5],
          [1.8, 12, 1.333333],
          [0.111, 0.2, 0.12],
          [1.111, 1.2, 1.12],
          [1.6, -1.5, 1.5]],
         [[1.1, -1.4, 1.6],
          [1.4, 0.0, 1.7],
          [0.111, 0.2, 0.12],
          [1.5, -1.8, 1.75],
          [1.5, 1.6, 1.6]],
         [[1.3, 1.9, 1.09],
          [0.111, 0.2, 0.12],
          [1.9, 1.4, 1.8],
          [1.6, 1.3, 1.0],
          [1.2, 1.4, 1.6]]],
        ])
    # print(inp.shape)

    expected_output = torch.tensor(
        [[[1.0428, 1.0231, 2.7393],
          [0.3979, 0.2152, 0.4686],
          [0.0000, 0.1198, 0.6974]],
         [[0.5006, 1.6137, 3.2702],
          [0.4512, 0.6272, 0.9556],
          [0.0000, 0.1232, 1.1280]]])
    verify_sol(inp, expected_output)

    verify_sol(torch.zeros_like(inp), torch.tensor(
        [[[0.1984, 0.0000, 0.2634],
          [0.1984, 0.0000, 0.2634],
          [0.1984, 0.0000, 0.2634]],
         [[0.1984, 0.0000, 0.2634],
          [0.1984, 0.0000, 0.2634],
          [0.1984, 0.0000, 0.2634]]]))

    print("All Sanity Checks Passed for Question 1h: highway.Highway!")
    print ("-"*80)


def run_checkers(args):
    checkers = {
        '1h': question_1h_sanity_check,
        '1i': question_1i_sanity_check,
    }

    checker_found = False
    for checker_name in checkers:
        if args[checker_name]:
            checker_found = True
            checkers[checker_name]()
    return checker_found

def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check Python & PyTorch Versions
    assert (sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ == "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if not run_checkers(args):
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
