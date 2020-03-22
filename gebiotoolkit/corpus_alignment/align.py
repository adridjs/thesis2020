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
from collections import defaultdict

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


def remove_tmp(person, languages):
    """
    Removes the temporary file created by extract in embed_extractor.py
    :param languages: Languages in which the temporary file has been created
    :type languages: set
    """
    for lang in languages:
        os.remove(f'tmp_preprocess/{lang}/{person}')
        os.remove(f'embeds/{lang}/{person}')


def run(encoder, person, person_filenames, languages, threshold=1.1, source_language='en'):
    """
    Run an :param encoder: onto the given :param person_files: in the set of :param languages: passed. The threshold is used to filter out invalid
    sentence pairs.

    :param encoder:
    :param person:
    :param person_filenames:
    :param languages:
    :param threshold:
    :param source_language:
    :return:
    """
    bpe_codes = LASER + 'models/93langs.fcodes'
    output_file = f'{HOME}/thesis2020/gebiotoolkit/corpus_alignment/parallel.tmp'  # parallel sentences will be stored here
    tmp_preprocess_fn = f'{HOME}/thesis2020/gebiotoolkit/corpus_alignment/tmp_preprocess'
    tmp_embeds_fn = f'{HOME}/thesis2020/gebiotoolkit/corpus_alignment/embeds'
    all_embeds = []
    for lang in languages:
        print(f'Preprocessing files: {lang}/{person}')
        preprocess(f'{tmp_preprocess_fn}/{lang}/{" ".join(person.split("_"))}', person_filenames[lang])
        all_embeds.append(extract(encoder, lang, bpe_codes, f'{tmp_preprocess_fn}/{lang}/{person}', f'{tmp_embeds_fn}/{lang}/{person}', verbose=True))

    print(f'Preprocessing finished')
    candidate_sentences = defaultdict(list)
    try:
        idx = languages.index(source_language)
        languages.pop(idx)
    except ValueError:
        print(f'Trying to remove {source_language} from languages list, but it doesn\'t exist in it. Did you miss passing the source language?')
    for lang in languages:
        parallel_sentences = mine(f'{tmp_preprocess_fn}/{source_language}/{person}',
                                 f'{tmp_preprocess_fn}/{lang}/{person}',
                                  source_language, lang,
                                 f'{tmp_embeds_fn}/{source_language}/{person}', f'{tmp_embeds_fn}/{lang}/{person}',
                                  output_file, 'mine')
        for r_s_tuple in parallel_sentences:
            ratio, sentences = r_s_tuple[0], r_s_tuple[1:]
            if float(ratio) < threshold:
                break
            candidate_sentences[lang].append(sentences)

    languages.append(source_language)
    remove_tmp(person, languages)
    return candidate_sentences


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
    parser = argparse.ArgumentParser(description='Stores sentences of persons appearing in all languages given by --languages command')
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

    source_language = 'en'
    encoder = generate_encoder(encoder_file)
    names = get_names_in_all_languages(corpus_folder, languages)
    names = list(map(lambda s: '_'.join(s.split()), names))
    for person in names:
        print(person)
        person_filenames = get_person_filenames_by_language(corpus_folder, person, languages=languages)
        sentences = run(encoder, person, person_filenames, languages, source_language=source_language)
        if sentences:
            store_sentences(sentences, person_filenames['en'], results_folder, person, source_language='en')


if __name__ == '__main__':
    main()
