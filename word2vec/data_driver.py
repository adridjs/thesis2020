import argparse
import random
import re
from collections import defaultdict
import sys

import spacy

sys.path.append(".")
from utils.constants import GENDERS, LANGUAGES, STOP_WORDS, NLP_MODELS
from utils.regexp import RegExp
from gebiotoolkit.storage_modules.file_restructure import include_sentence, store_sentences


class DataDriver:
    """
    :param corpus_folder: Path to the folder where the aligned corpus is stored
    :type corpus_folder: str
    :param languages: Languages from which to extract data from :param corpus_folder:
    :type languages: set
    :param genders: Genders from which to extract data from :param corpus_folder:
    :type genders: set
    """
    def __init__(self, corpus_folder, languages=None, genders=None):
        self.corpus_folder = corpus_folder
        self.languages = languages or LANGUAGES
        self.genders = genders or GENDERS
        self.re = RegExp()
        self.nlp_model_mapping = NLP_MODELS

    @staticmethod
    def _clean_sentence(model, txt):
        return model(txt).doc.text_with_ws.split()

    def _load_model(self, language):
        """
        Load spacy model based on :param language. This was created in order to have only one model loaded at a time, as some spacy models are huge.
        :param language: The language in which to retrieve the spacy's model name.
        :type language: str
        :return:
        """
        try:
            model_name = self.nlp_model_mapping[language]
        except KeyError:
            raise ValueError(f'Could not get a mapping between the language specified {language} and a model name. Check that you have it set at'
                             f' NLP_MODELS constant.')

        return spacy.load(model_name)

    def _get_filenames(self):
        """
        Helper function to retrieve input and output filenames based on :param languages and :param genders key sets.
        :return: An iterator of triples following (key, filename, out_filename)
        :rtype: tuple
        """

        for lang in self.languages:
            for gender in GENDERS:
                key = f'{lang}_{gender}'
                filename = f'{self.corpus_folder}/{key}.txt'
                out_filename = f'{self.corpus_folder}/{key}.filtered.txt'
                yield key, filename, out_filename

    def _parse_filtered_docs(self):
        """
        Get the documents for each lang-gender key and return them together with the key that has the least documents
        :return: The documents and the key with least documents
        :rtype: dict[str, list]
        """
        docs = defaultdict(list)
        least_docs_key = (None, 10 ** 6)
        for lang in self.languages:
            for gender in self.genders:
                key = f'{lang}_{gender}'
                filename = f'{self.corpus_folder}/{key}.filtered.txt'
                with open(filename, 'r') as f:
                    lines = ' '.join(f.readlines())
                    for n, doc in enumerate(re.finditer(self.re.doc_wise, lines, re.UNICODE)):
                        name, sentence = doc.groups()
                        docs[key].append(eval(sentence))

                    n_docs = len(docs[key])
                    if n_docs < least_docs_key[1]:
                        least_docs_key = (gender, n_docs)

        return docs, least_docs_key

    def _balance_dataset(self, docs, least_docs_value):
        """
        Balances a dataset composed by :param docs by randomly deleting samples from a given key until it is equal to :param least_docs_value
        :param docs: Documents in the data set
        :type docs: dict[str, list]
        :param least_docs_value: least number of documents in the whole data set
        :return:
        """
        balanced_dataset = defaultdict(list)
        for lang in self.languages:
            for gender in GENDERS:
                key = f'{lang}_{gender}'
                length = len(docs[key])
                n_delete = length - least_docs_value
                if n_delete == 0:
                    balanced_dataset[key] = docs[key]
                    continue

                print(f'Number of docs for gender: {gender} in language: {lang} -> {length}')
                print(f'Randomly deleting {n_delete} documents to have a uniform distribution between genders: {lang} -> {length}')
                sample = random.sample(docs[key], least_docs_value)
                balanced_dataset[key] = sample

        return balanced_dataset

    def get_balanced_dataset(self):
        """
        Retrieves documents matching the :param languages and :param genders key sets.
        :return:
        """
        # Get docs to quantify how many he/she instances exist in each language.
        docs, (least_docs_key, least_docs_value) = self._parse_filtered_docs()
        print(f'Key with least documents ({least_docs_value}): {least_docs_key}')

        # Balance dataset based on least_docs_val.
        dataset = self._balance_dataset(docs, least_docs_value=least_docs_value)

        return dataset

    def save_xml(self):
        """
        Returns a list of a list of words for the specified file names constructed by :param corpus_folder and :param languages
        Each sublist is a sentence.
        :return:
        """
        for key, file_name, out_file_name in self._get_filenames():
            lang, gender = file_name.split('/')[-1].split('_')
            gender = gender.split('.')[0]
            model = self._load_model(lang)
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

                person_sentence_line = self._clean_sentence(model, line)
                if len(person_sentence_line) > 1:
                    person_sentences_list.append(person_sentence_line)

            out_file.close()


def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f,""--corpus_folder", dest='corpus_folder', help="paths to corpus data", default='biographies')
    parser.add_argument("-l", "--languages", dest='languages', nargs='+', help="white-spaced languages you want to process the data on")

    return parser.parse_args()


def main():
    args = retrieve_args()

    corpus_folder = args.corpus_folder
    languages = args.languages

    dd = DataDriver(corpus_folder, languages=languages)
    # Generate files *.{filtered}.txt
    dd.save_xml()
    dd.get_balanced_dataset()


if __name__ == '__main__':
    main()
