#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 21:24:01 2019

@author: plxiv
"""

from nltk.tokenize import word_tokenize


def store_sentences(sentences, en_doc, results_folder, person, source_language='en'):
    written_src = False
    for target_lang, src_targ in sentences.items():
        gender = find_pronouns(en_doc)
        source_language_s, target_language_s = list(zip(*src_targ))
        for trg_s in target_language_s:
            with open(f'{results_folder}/{target_lang}_{gender}.txt', 'a') as f:
                if '\n' in src_targ:
                    f.write(person + ' : ' + trg_s)
                else:
                    f.write(person + ' : ' + trg_s + '\n')

        if not written_src:
            for src_s in source_language_s:
                with open(f'{results_folder}/{source_language}_{gender}.txt', 'a') as f:
                    if '\n' in src_targ:
                        f.write(person + ' : ' + src_s)
                    else:
                        f.write(person + ' : ' + src_s + '\n')
            written_src = True

def find_pronouns(filename):
    a = open(filename, 'r')
    text = a.readlines()
    text = [i for i in text if '\n' != i]
    concat_text = ' '.join(text[1:]).lower()
    tokens = word_tokenize(concat_text)
    he = len(list(filter(lambda x: x=='he' , tokens)))
    his = len(list(filter(lambda x: x=='his' , tokens)))
    she = len(list(filter(lambda x: x=='she' , tokens)))
    her = len(list(filter(lambda x: x=='her' , tokens)))
    if (he + his) > (she + her):
        gender = 'he'
    else:
        gender = 'she'
    return gender
