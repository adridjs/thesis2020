import spacy
import re

from word2vec.data_driver import DataDriver
from word2vec.utils.constants import NLP_MODELS

import json
from _collections import defaultdict


def get_lemma2words(lookup_folder, languages):
    """
    TODO
    :param lookup_folder:
    :param languages:
    :return:
    """
    l2w_by_language = dict()
    for lang in languages:
        fn = f'{lookup_folder}/{lang}_lemma_lookup.json'
        word2lemma = json.load(open(fn))
        lemma2words = defaultdict(list)
        for word, lemma in word2lemma.items():
            lemma2words[lemma].append(word)
        l2w_by_language[lang] = lemma2words
    return l2w_by_language


def get_gendered_words(docs):
    """
    TODO
    :param docs:
    :return:
    """
    gendered_words = dict()
    for lang_gender, sentences_by_lang in docs.items():
        lang, _ = lang_gender.split('_')
        model = spacy.load(NLP_MODELS[lang])
        print(f'Processing language and gender: {lang_gender}')
        gendered_words[lang_gender] = set()
        for sentence in sentences_by_lang:
            doc = model(' '.join(sentence))
            for token in doc:
                tag = token.tag_
                pos = token.pos_
                m = re.match(r'.*Gender=(\w*)', tag)
                if m and pos in ['ADJ', 'DET', 'NOUN']:
                    # gender = m.groups()
                    gendered_words[lang_gender].add((token.text, token.lemma_, pos))
                    # TODO:
    return gendered_words


def get_replace_word_es(word, lemma, candidates):
    """
    TODO
    :param word:
    :param lemma:
    :param candidates:
    :return:
    """
    replace = None
    if len(candidates) == 1:
        print(f'WARNING: Found one-to-one mapping between {candidates[0]} and {lemma}')
    elif word.endswith('as'):
        if lemma + 's' in candidates:
            replace = lemma + 's'
    elif word.endswith('os'):
        root = lemma[:-1]
        if root + 'as' in candidates:
            replace = root + 'as'
    elif word.endswith('a'):
        replace = lemma
    elif word.endswith('o'):
        root = lemma[:-1]
        if root + 'a' in candidates:
            replace = root + 'a'
    return replace


def replace_words_mapping(words_to_replace, language, l2w):
    """
    TODO
    :param words_to_replace:
    :param language:
    :param l2w:
    :return:
    """
    replace_mapping = dict()
    for triplet in words_to_replace:
        # lemma from spacy is actually the masculine and singular form
        word, lemma, pos = triplet
        # we want to change gender to all words independently of gender
        if language == 'es':
            candidates = l2w[lemma]  # if mapping is not one-to-one, it may contain the opposite gender word
            replace = get_replace_word_es(word, lemma, candidates)
            if replace:
                replace_mapping.update({word: replace})
        elif language == 'en':
            # TODO
            # get_replace_word_en
            pass

    return replace_mapping


if __name__ == '__main__':
    languages = {'es', 'en'}
    f = '/home/johndoe/.envs/thesis/lib/python3.6/site-packages/spacy_lookups_data/data/'
    l2w = get_lemma2words(f, languages)
    dd = DataDriver('../word2vec/biographies/', languages=languages)
    docs, _ = dd._parse_filtered_docs()
    # TODO: Insert this DataDriver.get_balanced_dataset() in order to have a uniform distribution of sentences between genders.
    gend_words = get_gendered_words(docs)
    for lang_gender, words in gend_words.items():
        lang, gender = lang_gender.split('_')
        # for each word in words, check if we have the word lemma in the lookup table
        words_to_replace = list(filter(lambda x: x[1] in l2w[lang], words))
        # we get the opposite gender word
        word2replace_mapping = replace_words_mapping(words_to_replace, lang, l2w)
        # TODO: Actually replace the words in the documents.


# In practice, however, using individual words as the unit of comparison is not optimal. Instead, BLEU computes the same modified precision
# metric using n-grams. The length which has the "highest correlation with monolingual human judgements"[5] was found to be four. The
# unigram scores are found to account for the adequacy of the translation, how much information is retained. The longer n-gram scores
# account for the fluency of the translation, or to what extent it reads like "good English".
