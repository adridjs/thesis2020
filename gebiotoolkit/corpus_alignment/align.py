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


def get_person_filenames_by_language(corpus_folder, person, languages):
    """
    Retrieves document filenames by language, given an input folder :param corpus_folder:, a :param person: name and a set of :param languages:
    :param corpus_folder: Folder in which to look for documents
    :type corpus_folder: str
    :param person: Person to look for
    :type person: str
    :param languages: Languages to look for
    :type languages: set
    :return: Dictionary of file names where its keys is the language and its values is the filename string
    :rtype: dict[str, str]
    """
    person_filenames = {}
    for lan in languages:
        person_filenames[lan] = corpus_folder + '/' + lan + '/raw/' + person
    return person_filenames


def remove_tmp(languages):
    """
    Removes the temporary file created by extract in embed_extractor.py
    :param languages: Languages in which the temporary file has been created
    :type languages: set
    :return:
    """
    for lang in languages:
        try:
            os.remove('tmp_preprocess/' + lang)
            os.remove('embeds/' + lang)
        except:
            pass


def extract_candidate_sentences(languages, person_filenames, encoder, threshold):
    """

    :param languages:
    :param person_filenames:
    :param encoder:
    :param threshold:
    :return:
    """
    bpe_codes = LASER + 'models/93langs.fcodes'
    output_file = f'{HOME}/thesis2020/gebiotoolkit/corpus_alignment/parallel.tmp'  # parallel sentences will be stored here
    tmp_preprocess_fn = f'{HOME}/thesis2020/gebiotoolkit/corpus_alignment/tmp_preprocess'
    tmp_embeds_fn = f'{HOME}/thesis2020/gebiotoolkit/corpus_alignment/embeds'
    candidate_sentences = []
    all_embeds = []
    for lan in languages:
        print(f'Preprocessing files: {lan}')
        preprocess(f'{tmp_preprocess_fn}/{lan}', person_filenames[lan])
        all_embeds.append(extract(encoder, lan, bpe_codes, f'{tmp_preprocess_fn}/{lan}', f'{tmp_embeds_fn}/{lan}', verbose=True))

    print(f'Preprocessing finished')
    for lan in languages:
        parallel_sentences = mine(f'{tmp_preprocess_fn}/{languages[0]}',
                                 f'{tmp_preprocess_fn}/{lan}',
                                 languages[0], lan,
                                 f'{tmp_embeds_fn}/{languages[0]}', f'{tmp_embeds_fn}/{lan}',
                                 output_file, 'mine')
        for parallel_sentence in parallel_sentences:
            if float(parallel_sentence[0]) > threshold:
                candidate_sentences.append(parallel_sentence)
            else:
                break

    remove_tmp(languages)
    return candidate_sentences


def compare_sentences(lan_1, lan_2):
    """
    Given a list of sentences in
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


def find_parallel_sentences(candidate_sentences):
    """
    Given a list of :param candidate_sentences: compare them and extract valid parallel sentences.
    :param candidate_sentences: Sentences that we want to check are valid parallel sentences.
    :type candidate_sentences: list of tuple
    :return: A list of parallel sentences, which is a subset of the given :param candidate_sentences
    :rtype: list of tuple
    """
    try:
        parallel_sentences = [[i[1]] for i in candidate_sentences[0]]
        for sentence in candidate_sentences:
            parallel_sentences = compare_sentences(parallel_sentences, sentence)
    except:
        parallel_sentences = []
    return parallel_sentences


def run(encoder, person_filenames, languages, threshold=1.055):
    """
    Run an :param encoder: onto the given :param person_files: in the set of :param languages: passed. The threshold is used to filter out invalid
    sentence pairs.
    :param encoder:
    :param person_filenames:
    :param languages: Languages in which the encoder will be passed.
    :param threshold: Ratio between pairs of sentences to be accepted as valid.
    :return:
    """
    candidate_sentences = extract_candidate_sentences(languages, person_filenames, encoder, threshold)
    parallel_sentences = find_parallel_sentences(candidate_sentences)
    return parallel_sentences


def get_names_in_all_languages(corpus_folder, languages):
    """
    Given a :param corpus_folder: and a set of languages, retrieve persons whose documents exist in all :param languages:
    :param corpus_folder: Folder where the processed wiki dump files are saved. Normally, it follows /wiki/{lang}/xxxx , i.e lang  'es'
    :type corpus_folder: str
    :param languages: Set of languages in which to look for existing documents.
    :type languages: set
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
        person_filenames = get_person_filenames_by_language(corpus_folder, person, languages=languages)
        sentences = run(encoder, person_filenames, languages)
        en_doc = open(person_filenames['en']).readlines()
        if sentences:
            store_sentences(sentences, en_doc, languages, results_folder, person)


if __name__ == '__main__':
    main()
