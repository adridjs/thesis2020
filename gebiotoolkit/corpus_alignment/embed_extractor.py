#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 20:37:13 2019

@author: plxiv
"""

import os
import sys
import tempfile

home = os.environ['HOME']
os.environ['LASER'] = f'{home}/LASER/'
LASER = os.environ['LASER']
sys.path.append(LASER + 'source/')
sys.path.append(LASER + '/source/lib')

from text_processing import Token, BPEfastApply
from embed import SentenceEncoder, EncodeFile, EmbedLoad


def extract(encoder, token_lang, bpe_codes, ifname, output, verbose=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        print(tmpdir)

        if token_lang != '--':
            tok_fname = os.path.join(tmpdir, 'tok')
            romanize = True if token_lang == 'el' else False
            Token(ifname, tok_fname, lang=token_lang, romanize=romanize, lower_case=True, gzip=False,  verbose=verbose, over_write=True)
            ifname = tok_fname

        if bpe_codes:
            bpe_fname = os.path.join(tmpdir, 'bpe')
            BPEfastApply(ifname, bpe_fname, bpe_codes, verbose=verbose, over_write=True)
            ifname = bpe_fname

        EncodeFile(encoder, ifname, output, verbose=verbose, over_write=True, buffer_size=10000)
        return EmbedLoad(output)


def generate_encoder(encoder_file):
    return SentenceEncoder(encoder_file, max_sentences=None, max_tokens=12000, sort_kind='quicksort', cpu=True)
