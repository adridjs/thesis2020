import random
import re
from _collections import defaultdict

from utils.regexp import RegExp
from utils.constants import GENDERS, LANGUAGES

_re = RegExp()


def get_docs(folder='biographies/', languages=None):
    docs = defaultdict(list)
    least_docs_key = (None, 10**6)
    for lang in languages:
        for gender in GENDERS:
            key = f'{lang}_{gender}'
            filename = f'{folder}{key}.filtered.txt'
            with open(filename, 'r') as f:
                lines = ' '.join(f.readlines())
                for n, doc in enumerate(re.finditer(_re.doc_wise, lines, re.UNICODE)):
                    name, sentence = doc.groups()
                    if name is None:
                        sentences.append(eval(sentence))
                    elif n != 0:
                        docs[key].append(sentences)
                    sentences = [eval(sentence)]

                n_docs = len(docs[key])
                if n_docs < least_docs_key[1]:
                    least_docs_key = (gender, n_docs)

    return docs, least_docs_key


def balance_dataset(docs, least_docs_value, languages=None):
    balanced_dataset = defaultdict(list)
    for lang in languages:
        for gender in GENDERS:
            key = f'{lang}_{gender}'
            length = len(docs[key])
            n_delete = length - least_docs_value
            if n_delete == 0:
                balanced_dataset = docs[key]
                continue

            print(f'Number of docs for gender: {gender} in language: {lang} -> {length}')
            print(f'Randomly deleting {n_delete} documents to have a uniform distribution between genders: {lang} -> {length}')
            sample = random.sample(docs[key], least_docs_value)
            balanced_dataset[key] = sample

    return balanced_dataset


def get_balanced_dataset(folder='biographies/', languages=None):
    if not languages:
        languages = LANGUAGES

    # Get docs to quantify how many he/she instances exist in each language.
    docs, (least_docs_key, least_docs_value) = get_docs(folder, languages)
    print(f'Key with least documents ({least_docs_value}): {least_docs_key}')

    # Balance dataset based on least_docs_val.
    dataset = balance_dataset(docs, least_docs_value=least_docs_value, languages=languages)

    return dataset


if __name__ == '__main__':
    get_balanced_dataset(languages={'en'})


