#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 23:34:57 2019

@author: plxiv
"""
import os
import subprocess 

home = os.environ['HOME']
os.environ['LASER'] = f'{home}/LASER/'
LASER = os.environ['LASER']


def mine(src, trg, src_lang, trg_lang, src_embeddings, trg_embeddings, output, mode):
    mine_file = LASER + 'source/mine_bitexts.py'
    src = src.replace("'", "\'")
    trg = trg.replace("'", "\'")
    src_embeddings = src_embeddings.replace("'", "\'")
    trg_embeddings = trg_embeddings.replace("'", "\'")
    print(f'Mine file: {mine_file} \n'
          f'src: {src}\n trg: {trg}\n src-lang: {src_lang}\n trg-lang: {trg_lang}\n src-embeddings {src_embeddings} \n'
                          f'trg-embeddings {trg_embeddings}\noutput {output}\nmode {mode}')
    command = ['python3', f'{mine_file}', f'{src}', f'{trg}', '--src-lang', f'{src_lang}', '--trg-lang', f'{trg_lang}', '--mode', f'{mode}', '--verbose']
    command += ['--src-embeddings', f'{src_embeddings}', '--trg-embeddings', f'{trg_embeddings}', '--output', f'{output}']
    print(command)
    subprocess.run(command)
    with open(output, 'r', encoding='utf8') as f:
        parallel = f.readlines()
    for i, j in enumerate(parallel):
        parallel[i] = j.split('\t')
    os.remove(output)
    return parallel
