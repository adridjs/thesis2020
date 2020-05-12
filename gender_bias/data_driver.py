import argparse
import random
import re
from collections import defaultdict
import sys

import spacy

sys.path.append(".")
from gender_bias.utils.constants import GENDERS, LANGUAGES, NLP_MODELS
from gender_bias.utils.regexp import RegExp
from gebiotoolkit.storage_modules.file_restructure import parse_sentence, save_xml


class DataDriver:
    """
    :param corpus_folder: Path to the folder where the aligned corpus is stored
    :type corpus_folder: str
    :param
    :param languages: Languages from which to extract data from :param corpus_folder:
    :type languages: set
    :param genders: Genders from which to extract data from :param corpus_folder:
    :type genders: set
    """
    def __init__(self, corpus_folder, save_dir, languages=None, genders=None):
        self.corpus_folder = corpus_folder
        self.save_dir= save_dir
        self.languages = languages or LANGUAGES
        self.genders = genders or GENDERS
        self.re = RegExp()
        self.nlp_model_mapping = NLP_MODELS

    @staticmethod
    def _clean_sentence(txt):
        pattern = r'<(a|/a).*?>'
        result = re.sub(pattern, "", txt)
        return result

    def _load_model(self, language):
        """
        Load spacy model based on :param language. This was created in order to have only one model loaded at a time, as some spacy models are huge.
        :param language: The language in which to retrieve the spacy's model name.
        :type language: str
        :return: TODO
        """
        try:
            model_name = self.nlp_model_mapping[language]
        except KeyError:
            raise ValueError(f'Could not get a mapping between the language specified {language} and a model name. Check that you have it set at'
                             f' NLP_MODELS constant.')

        return spacy.load(model_name)

    def _get_gender_filenames(self, format='xml'):
        """
        Helper function to retrieve input and output filenames based on :param languages and :param genders key sets.
        :return: An iterator of triples following (key, filename, out_filename)
        :rtype: tuple
        """

        for lang in self.languages:
            for gender in GENDERS:
                key = (lang, gender)
                filename = f'{self.corpus_folder}/{key}.txt'
                out_filename = f'{self.save_dir}/{key}.balanced.{format}'
                yield key, filename, out_filename

    def _parse_biographies(self, format='xml'):
        """
        Get the documents for each lang-gender key and return them together with the key that has the least documents
        :return: The documents as a dictionary and the key with least documents along with its value as a tuple
        :rtype: dict[str, list], tuple[str, int]
        """
        docs = defaultdict(list)
        least_docs_key = (None, 10 ** 6)
        for lang in self.languages:
            nlp_model = self._load_model(language=lang)
            for gender in self.genders:
                key = f'{lang}_{gender}'
                filename = f'{self.corpus_folder}/{key}.filtered.{format}'
                with open(filename, 'r') as f:
                    if format == 'xml':
                        lines = ' '.join(f.readlines())
                        for n, doc in enumerate(re.finditer(self.re.doc_wise, lines, re.UNICODE)):
                            name, sentence = doc.groups()
                            docs[key].append([token.text for token in nlp_model(sentence)])
                    else:
                        docs[key] = f.readlines()

                    n_docs = len(docs[key])
                    if n_docs < least_docs_key[1]:
                        least_docs_key = (gender, n_docs)

        return docs, least_docs_key

    def _balance_dataset(self, docs, least_docs_value, seed=15):
        """
        Balances a dataset composed by :param docs by randomly deleting samples from a given key until it is equal to :param least_docs_value
        :param docs: Documents in the data set
        :type docs: dict[str, list]
        :param least_docs_value: least number of documents in the whole data set
        :type
        """
        random.seed(seed)
        self.balanced_dataset = defaultdict(list)
        for lang in self.languages:
            for gender in GENDERS:
                key = f'{lang}_{gender}'
                length = len(docs[key])
                n_delete = length - least_docs_value
                if n_delete != 0:
                    print(f'Number of docs for gender: {gender} in language: {lang} -> {length}. Randomly deleting {n_delete} '
                          f'documents to have a uniform distribution between genders')
                    sample = random.sample(docs[key], least_docs_value)  # For reproducibility purpose
                    self.balanced_dataset[key] = sample
                else:
                    self.balanced_dataset[key] = docs[key]

    def get_balanced_dataset(self):
        """
        Retrieves documents matching the :param languages and :param genders key sets.
        """
        # Get docs to quantify how many he/she instances exist in each language.
        docs, (least_docs_key, least_docs_value) = self._parse_biographies(format='txt')
        print(f'Key with least documents ({least_docs_value}): {least_docs_key}')

        # Balance dataset based on least_docs_val.
        self._balance_dataset(docs, least_docs_value=least_docs_value)

    def save_sentences(self, format='xml'):
        """
        Saves a list of a list of words for the specified file names constructed by :param corpus_folder and :param languages
        Each sublist is a sentence.
        :return:
        """
        if not self.balanced_dataset:
            for key, filename, out_filename in self._get_gender_filenames(format=format):
                lang, gender = key
                gender = gender.split('.')[0]
                with open(out_filename, 'w+') as out_file:
                    sentences_by_person = self.process_file(filename)
                    for name, sentences in sentences_by_person.items():
                        if format == 'txt':
                            out_file.write('\n'.join(sentences))
                        elif format == 'xml':
                            save_xml(out_file, name, sentences, lang, gender)
        else:
            for key, sentences in self.balanced_dataset.items():
                with open(f'{self.save_dir}/{key}.balanced.{format}', 'w+') as out_file:
                    out_file.writelines(sentences)

    def process_file(self, filename):
        sentences_by_person = defaultdict(list)
        for line in open(filename):
            line = line.strip().lower()
            sentence, name = parse_sentence(line)
            cleaned_sentence = self._clean_sentence(sentence)
            sentences_by_person[name].append(cleaned_sentence)

        return sentences_by_person


def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f,""--corpus_folder", dest='corpus_folder', help="paths to corpus data",
                        default='../gebiotoolkit/corpus_alignment/aligned')
    parser.add_argument("-s,""--save_dir", dest='save_dir', help="paths where the generated dataset will be saved",
                        default='../translation/domain_adaptation/data')
    parser.add_argument("-l", "--languages", dest='languages', nargs='+', help="white-spaced languages you want to process the data on")

    return parser.parse_args()


def main():
    args = retrieve_args()

    corpus_folder = args.corpus_folder
    save_dir = args.save_dir
    languages = args.languages
    dd = DataDriver(corpus_folder, save_dir, languages=languages)
    dd.get_balanced_dataset()
    # Generate files *.filtered.txt
    dd.save_sentences(format='txt')


if __name__ == '__main__':
    main()
