import re
import argparse
from collections import defaultdict

from nltk.corpus import stopwords

from gebiotoolkit.storage_modules.file_restructure import include_sentence, store_sentences

STOP_WORDS = set(stopwords.words('english'))
GENDERS = {'he', 'she'}


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
    sentence_words = defaultdict(list)
    for file_name, out_file_name in zip(*_get_filenames(folder, languages=languages)):
        lang, gender = file_name.split('/')[-1].split('_')
        gender = gender.split('.')[0]

        out_file = open(out_file_name, 'w+')
        person_sentences_list = list()
        for n, line in enumerate(open(file_name)):
            line = line.strip().lower()
            line, current_name = include_sentence(line)
            if n == 0:
                last_name = current_name

            if current_name != last_name:
                store_sentences(out_file, last_name, person_sentences_list, lang, gender)
                last_name = current_name
                person_sentences_list = list()

            person_sentence_line = get_words(line)
            if len(person_sentence_line) > 1:
                person_sentences_list.append(person_sentence_line)
                sentence_words[f'{lang}-{gender}'].append((current_name, person_sentence_line))

        out_file.close()

    print(f'Unique sentences for each combination of {languages} x {GENDERS} ')
    return sentence_words


def _get_filenames(folder, languages):
    filenames = []
    out_filenames = []
    filenames.extend([f'{folder}/{lang}_{gender}.txt' for lang in languages for gender in GENDERS])
    out_filenames.extend([f'{folder}/{lang}_{gender}.filtered.txt' for lang in languages for gender in GENDERS])

    return filenames, out_filenames


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

"""In order to process files from folder 'word2vec/biographies' -l en es"""
