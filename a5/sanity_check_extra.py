#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
sanity_check.py: sanity checks for assignment 5
Usage:
    sanity_check.py 1h
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


class DummyVocab():
    def __init__(self):
        self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]

def question_1h_sanity_check():
    """ Sanity check for highway.Highway module.
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1h: highway.Highway module")
    print ("-"*80)

    dim = 3
    highway = Highway(dim)

    assert highway.proj.weight.shape == (dim, dim), \
            'Expected proj to have weight matrix of shape (%s, %s)' % (dim, dim)
    assert highway.proj.bias.shape == (dim, dim), \
            'Expected proj to have bias vector of shape (%s)' % (dim)
    assert highway.gate.weight.shape == (dim, dim), \
            'Expected gate to have weight matrix of shape (%s, %s)' % (dim, dim)
    assert highway.gate.bias.shape == (dim, dim), \
            'Expected gate to have bias vector of shape (%s)' % (dim)

    print("All Sanity Checks Passed for Question 1e: words2charindices()!")
    print ("-"*80)

def run_checkers():
    checkers = {
        '1h': question_1h_sanity_check
    }

    checker_found = False
    for (arg_name, checker) in checkers:
        if args[arg_name]:
            checker_found = True
            checker()
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

    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

    # Create NMT Model
    model = NMT(
            embed_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            dropout_rate=DROPOUT_RATE,
            vocab=vocab)

    char_vocab = DummyVocab()

    # Initialize CharDecoder
    decoder = CharDecoder(
            hidden_size=HIDDEN_SIZE,
            char_embedding_size=EMBED_SIZE,
            target_vocab=char_vocab)

    if not run_chekcers():
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
