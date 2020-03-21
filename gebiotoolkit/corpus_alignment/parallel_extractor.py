#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 23:34:57 2019

@author: plxiv
"""
import os
home = os.environ['HOME']
os.environ['LASER'] = f'{home}/LASER/'
LASER = os.environ['LASER']


def mine(src, trg, src_lang, trg_lang, src_embeddings, trg_embeddings, output, mode):
    mine_file = LASER + 'source/mine_bitexts.py'
    print(f'Mine file: {mine_file} \n'
          f'src: {src}\n trg: {trg}\n src-lang: {src_lang}\n trg-lang: {trg_lang}\n src-embeddings {src_embeddings} \n'
                          f'trg-embeddings {trg_embeddings}\noutput {output}\nmode {mode}')
    command = f'python3 {mine_file} {src} {trg} --src-lang {src_lang} --trg-lang {trg_lang} --src-embeddings {src_embeddings} ' \
                          f'--trg-embeddings {trg_embeddings} --output {output} --mode {mode} --verbose'
    try:
        os.system(command)
    except Exception as e:
        raise e

    with open(output, 'r', encoding='utf8') as f:
        parallel = f.readlines()
    for i, j in enumerate(parallel):
        parallel[i] = j.split('\t')
    os.remove(output)
    return parallel
