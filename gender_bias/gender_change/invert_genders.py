import spacy
import re
import logging
import json
from _collections import defaultdict

from gender_bias.DataDriver import DataDriver
from utils import NLP_MODELS

logging.basicConfig(filename='gender_change.log',level=logging.DEBUG)


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
        lang, gender = lang_gender.split('_')
        model = spacy.load(NLP_MODELS[lang])
        logging.info(f'Processing language and gender: {lang_gender}')
        gendered_words[lang_gender] = set()
        for sentence in sentences_by_lang:
            doc = model(' '.join(sentence))
            for token in doc:
                tag = token.tag_
                pos = token.pos_
                if pos in ['DET']:
                    m = re.match(r'.*Gender=(\w*)', tag)
                    if m:
                        gendered_words[lang_gender].add((token.text, token.lemma_, pos))

    return gendered_words


def replace_words_in_docs(docs, mapping):
    """

    :param docs:
    :param mapping:
    :return:
    """
    # escape words to be replaced
    replace_mapping = dict((re.escape(k), v) for k, v in mapping.items())
    # join them by | to consider all words in a single regexp
    pattern = re.compile("|".join(replace_mapping.keys()))
    changed_docs = list()
    for doc in docs:
        if any([wtr in doc for wtr in mapping.keys()]):
            # substitute the matched word by its mapping value
            changed_doc = pattern.sub(lambda m: mapping[m.group(0)], ' '.join(doc))
            changed_docs.append(changed_doc)
            logging.info(f'{doc} -----> {changed_doc}')
        else:
            # changed_docs.append(doc)
            pass
    return changed_docs


def get_gender_inverted_docs(docs, lemma2words):
    """

    :param docs:
    :param lemma2words:
    :return:
    """
    changed_docs = dict()
    gendered_words = get_gendered_words(docs)
    for lang_gender, words in gendered_words.items():
        docs_to_change = docs[lang_gender]
        lang, gender = lang_gender.split('_')
        logging.info(f'Number of gendered words in language {lang} and gender {gender}: {len(words)}')
        # for each word in words, check if we have the word lemma in the lookup table
        words_to_replace = list(filter(lambda x: x[1] in lemma2words[lang], words))
        logging.info(f'Trying to find a mapping for replacing a total of {len(words_to_replace)} words')
        # we get the opposite gender word
        replace_mapping = get_replace_words_mapping(words_to_replace, lang, lemma2words)
        logging.info(f'Found mapping for {len(words_to_replace)} words')
        changed_docs[lang_gender] = replace_words_in_docs(docs_to_change, replace_mapping)

    return changed_docs


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
        logging.info(f'WARNING: Found one-to-one mapping between {candidates[0]} and {lemma}')
    elif word.endswith('as') and lemma == word[:-1]:
        if lemma + 's' in candidates:
            replace = lemma + 's'
    elif word.endswith('os') and lemma == word[:-1]:
        root = lemma[:-1]
        if root + 'as' in candidates:
            replace = root + 'as'
    elif word.endswith('a') and lemma == word[:-1] + 'o':
        replace = lemma
    elif word.endswith('o'):
        root = lemma[:-1]
        if root + 'a' in candidates:
            replace = root + 'a'
    return replace


def get_replace_words_mapping(words_to_replace, language, lemma2words):
    """
    TODO
    :param words_to_replace:
    :param language:
    :param lemma2word:
    :return:
    """
    mapping = dict()
    for triplet in words_to_replace:
        # lemma from spacy is actually the masculine and singular form
        word, lemma, pos = triplet
        # we want to change gender to all words independently of gender
        if language == 'es':
            l2w = lemma2words.get(language)
            candidates = l2w.get(lemma)
            if len(candidates):  # if mapping is not one-to-one, it may contain the opposite gender word
                replace = get_replace_word_es(word, lemma, candidates)
                if replace:
                    logging.info(f'Replacing word {word} -> {replace}')
                    mapping.update({word: replace})
            else:
                logging.info(f'Couldn\'t find {lemma} from word {word} in l2w.')
        elif language == 'en':
            # TODO

            pass

    return mapping


def main():
    args = retrieve_args()
    languages = {'es'}
    lookup_fn = '/home/johndoe/.envs/thesis/lib/python3.6/site-packages/spacy_lookups_data/data/'
    l2w = get_lemma2words(lookup_fn, languages)
    dd = DataDriver('../gender_bias/biographies', languages=languages)
    docs, _ = dd.get_biographies_corpus()
    dd.get_balanced_corpus()
    gi_docs = get_gender_inverted_docs(docs, l2w)
    for lang_gender, docs in gi_docs.items():
        with open(f'{lang_gender}.inverted.txt', 'w+') as f:
            for doc in docs:
                f.write(doc + '\n')


if __name__ == '__main__':
    main()

# In practice, however, using individual words as the unit of comparison is not optimal. Instead, BLEU computes the same modified precision
# metric using n-grams. The length which has the "highest correlation with monolingual human judgements"[5] was found to be four. The
# unigram scores are found to account for the adequacy of the translation, how much information is retained. The longer n-gram scores
# account for the fluency of the translation, or to what extent it reads like "good English".
