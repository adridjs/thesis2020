import spacy
import re

from word2vec.data_driver import DataDriver
from word2vec.utils.constants import NLP_MODELS

import json
from _collections import defaultdict


def get_lemma2words(lookup_folder, languages):
    """

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
                if m and pos in ['ADJ', 'DET']:
                    # gender = m.groups()
                    gendered_words[lang_gender].add((token.text, token.lemma_))
                    # TODO:
    return gendered_words


if __name__ == '__main__':
    languages = {'es', 'en'}
    f = '/home/johndoe/.envs/thesis/lib/python3.6/site-packages/spacy_lookups_data/data/'
    l2w = get_lemma2words(f, languages)
    dd = DataDriver('../word2vec/biographies/', languages=languages)
    docs, _ = dd._parse_filtered_docs()
    # TODO: Insert this DataDriver.get_balanced_dataset() in order to have a uniform distribution of sentences between genders.

    gend_words = get_gendered_words(docs)

    words_to_change = dict()
    for lang_gender, words in gend_words:
        lang, gender = lang_gender.split('_')
        words_to_change[lang_gender] = list(filter(lambda x: x[1] in l2w[lang], words))
    pass

# In practice, however, using individual words as the unit of comparison is not optimal. Instead, BLEU computes the same modified precision
# metric using n-grams. The length which has the "highest correlation with monolingual human judgements"[5] was found to be four. The
# unigram scores are found to account for the adequacy of the translation, how much information is retained. The longer n-gram scores
# account for the fluency of the translation, or to what extent it reads like "good English".
