import re
import argparse
from collections import Counter

from nltk.corpus import stopwords

from gebiotoolkit.storage_modules.file_restructure import id_retriever, include_sentence, store_sentence

STOP_WORDS = set(stopwords.words('english'))
GENDERS = ['he', 'she']


def get_words(txt):
    return list(filter(
        lambda x: x not in STOP_WORDS,
        re.findall(r'\b(\w+)\b', txt)
    ))


def parse_sentence_words(folder, languages):
    """
    Returns a list of a list of words for the specified filenames constructed via :param folder: and :param languages:.
    Each sublist is a sentence.
    :param folder:
    :param languages:
    :return:
    """

    unique = Counter({f'{lang}-{gender}' for lang in languages for gender in GENDERS})
    sentences = Counter({f'{lang}-{gender}' for lang in languages for gender in GENDERS})

    sentence_words = []
    for file_name in _get_filenames(folder, languages=languages):
        lang, gender = file_name.split('/')[-1].split('_')
        gender = gender.split('.')[0]

        for line in open(file_name):
            line = line.strip().lower()
            line, name = include_sentence(line)
            wid = id_retriever(name, lang)
            sent_words = get_words(line)
            if len(sent_words) > 1:
                unique.update({f'{lang}-{gender}': 1})
                # store_sentence()
                sentence_words.append(sent_words)

    return sentence_words


def _get_filenames(folder, languages):

    filenames = []
    filenames.extend([f'{folder}/{lang}_{gender}.txt' for lang in languages for gender in GENDERS])

    return filenames


def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f,""--corpus_folder", dest='corpus_folder', help="paths to corpus data", default='../corpus_alignment/aligned/')
    parser.add_argument("-t", "--threshold", dest='threshold', help="threshold allowed ", default=1.5)
    parser.add_argument("-l", "--langs", dest='langs', nargs='+', help="langs in which the filter will be performed")

    return parser.parse_args()


def main():
    args = retrieve_args()
    corpus_folder = args.corpus_folder
    langs = args.langs

    biography_sentences = parse_sentence_words(corpus_folder, langs)
    print('\n'.join(map(str, biography_sentences[:10])))


if __name__ == '__main__':
    main()