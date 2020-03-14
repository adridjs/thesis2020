#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 02:03:01 2019
Edited on Sat March 14 23:47 2019 by @adridjs

@author: plxiv
"""
import os
import sys
import argparse

sys.path.append(os.getcwd())
from gebiotoolkit.storage_modules.storage_modules import store_sentences
from preprocess import preprocess
from embed_extractor import extract, generate_encoder
from parallel_extractor import mine

HOME = os.environ['HOME']
os.environ['LASER'] = f'{HOME}/LASER/'
LASER = os.environ['LASER']


def get_person_filenames_by_language(corpus_folder, person, list_languages):
    """

    :param corpus_folder:
    :param person:
    :param list_languages:
    :return:
    """
    person_filenames = {}
    for lan_p in list_languages:
        person_filenames[lan_p] = corpus_folder + '/' + lan_p + '/raw/' + person
    return person_filenames


def remove_tmp(languages):
    """

    :param languages:
    :return:
    """
    for i in languages:
        try:
            os.remove('tmp_preprocess/' + i)
            os.remove('embeds/' + i)
        except:
            pass


def extract_candidate_sentences(languages, person_filenames, encoder, bpe_codes, parallel_file, threshold):
    """

    :param languages:
    :param person_filenames:
    :param encoder:
    :param bpe_codes:
    :param parallel_file:
    :param threshold:
    :return:
    """
    sel_par_sentences = []
    all_embeds = []
    for lan in languages:
        preprocess(lan, person_filenames[lan])
        all_embeds.append(extract(encoder, lan, bpe_codes, 'tmp_preprocess/' + lan, 'embeds/' + str(lan), verbose = False))
    for lan in languages:
        all_par_sentences = mine('tmp_preprocess/' + languages[0],
                                 'tmp_preprocess/' + lan,
                                 languages[0], lan, 
                                 'embeds/' + languages[0], 'embeds/' + lan, 
                                 parallel_file, 'mine')
        if all_par_sentences:
            for i, par in enumerate(all_par_sentences):
                if float(par[0]) < threshold:
                    break
            sel_par_sentences.append(all_par_sentences[:i])
    remove_tmp(languages)
    return sel_par_sentences


def compare_sentences(lan_1, lan_2):
    """

    :param lan_1:
    :param lan_2:
    :return:
    """
    same_lan = [i[1] for i in lan_2]
    update = []
    for i, sentence in enumerate(lan_1):
        if sentence[0] in same_lan:
            lan_1[i].append(lan_2[same_lan.index(sentence[0])][2])
            update.append(lan_1[i])
    return update


def find_parallel_sentences(sel_par_sentences):
    """

    :param sel_par_sentences:
    :return:
    """
    try:
        parrallel_sentences = [[i[1]] for i in sel_par_sentences[0]]
        for i in range(1,len(sel_par_sentences)):
            parrallel_sentences = compare_sentences(parrallel_sentences, sel_par_sentences[i])
    except:
        parrallel_sentences = []
    return parrallel_sentences        


def run(encoder, person_filenames, languages, threshold=1.055):
    """
    Run an :param encoder: onto the given :param person_files: in the set of :param languages: passed. The threshold
    :param encoder:
    :param person_filenames:
    :param languages: Languages in which the encoder will be passed.
    :param threshold: Ratio between pairs of sentences to be accepted as valid.
    :return:
    """
    bpe_codes = LASER + 'models/93langs.fcodes'  
    parallel_tmp = 'parallel.txt'
    sel_par_sentences = extract_candidate_sentences(languages, person_filenames, encoder, bpe_codes, parallel_tmp, threshold)
    par = find_parallel_sentences(sel_par_sentences)
    return par


def get_names_in_all_languages(corpus_folder, languages):
    """
    Given a :param corpus_folder: and a set of languages, retrieve persons whose documents exist in all :param languages:
    :param corpus_folder: Folder where the processed wiki dump files are saved. Normally, it follows /wiki/{lang}/xxxx , i.e lang  'es'
    :param languages: Set of languages in which to look for existing documents.
    :return: Names that exist in all given :param languages:
    :rtype: set
    """
    names = dict()
    for lang in languages:
        names[lang] = list()
        for filename in os.listdir(f'{corpus_folder}/{lang}/raw/'):
            names[lang].append(filename)

    names_list = list(names.items())
    ordered_names = sorted(names_list, key=lambda x: len(x[1]), reverse=True)
    lang, max_names = ordered_names.pop(0)
    print(f'Language with maximum names -> {lang} ({len(max_names)})')
    names_all_langs = set([name for lang, names in ordered_names
                       for name in names if name in max_names])

    print(f'Names in all languages -> {len(names_all_langs)}')
    return names_all_langs


def retrieve_args():
    parser = argparse.ArgumentParser(description='Generates a pickle in which contains the dictionary of the samples in which all languages have '
                                                 'the same entry')
    parser.add_argument('-l', '--languages', nargs='+', required=True, help='Languages in which the parallel sentences will be generated')
    parser.add_argument('-f', '--folder', help='folder where the extracted corpus from wikipedia dumps is located',
                        default='gebiotoolkit/corpus_extraction/wiki')
    parser.add_argument('-s', '--save_path', required=False, help='Folder where the sentences will be stored', default='aligned/')
    parser.add_argument('-e', '--encoder', required=False, help='path to the LASER encoder',
                        default=f'{LASER}/models/bilstm.93langs.2018-12-26.pt')
    args = parser.parse_args()
    return args


def main():
    args = retrieve_args()
    corpus_folder = args.folder
    languages = args.languages
    results_folder = args.save_path
    encoder_file = args.encoder

    encoder = generate_encoder(encoder_file)
    names = get_names_in_all_languages(corpus_folder, languages)
    for person in names:
        print(person)
        person_filenames = get_person_filenames_by_language(corpus_folder, person, list_languages=languages)
        sentences = run(encoder, person_filenames, languages)
        if sentences:
            store_sentences(sentences, names, languages, results_folder, person)


if __name__ == '__main__':   
    main()
