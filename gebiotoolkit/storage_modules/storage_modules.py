#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 21:24:01 2019

@author: plxiv
"""

from nltk.tokenize import word_tokenize


def store_sentences(sentences, en_doc, languages, results_folder, person):
    for sentence in sentences:
        gender = find_pronouns(en_doc)
        if len(sentence) == len(languages):
            for language in languages:
                with open(f'{results_folder}/{language}_{gender}.txt', 'a') as f:
                    if '\n' in sentence[language]:
                        f.write(person + ' : ' + sentence[language])
                    else:
                        f.write(person + ' : ' + sentence[language] + '\n')


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


def load_wiki_names(wiki_filename):
    with open(wiki_filename) as f:
        sentence = f.readlines()
