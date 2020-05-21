import random
import re
from collections import defaultdict
import sys
import os

import spacy

sys.path.append(".")
from utils import GENDERS, LANGUAGES, NLP_MODELS
from utils import RegExp
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
    def _remove_anchor_tag(txt):
        pattern = r'<(a|/a).*?>'
        return re.sub(pattern, "", txt).strip()

    def _clean_sentence(self, txt):
        try:
            split = txt.strip().split(':')
            result = self._remove_anchor_tag(split)
            return result
        except (ValueError, IndexError):
            print(f'Error when splitting sentence by ":" {split}')

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

    def _get_biographies_corpus(self, format='xml'):
        """
        Get the documents for each lang-gender key and return them together with the key that has the least documents
        :return: The documents as a dictionary and the key with least documents along with its value as a tuple
        :rtype: dict[str, list], tuple[str, int]
        """
        biographies = defaultdict(list)
        least_docs_key = (None, 10 ** 6)
        for lang in self.languages:
            # nlp_model = self._load_model(language=lang)
            for gender in self.genders:
                key = f'{lang}_{gender}'
                filename = f'{self.corpus_folder}/{key}.{format}'
                with open(filename, 'r') as f:
                    if format == 'xml':
                        lines = ' '.join(f.readlines())
                        for n, doc in enumerate(re.finditer(self.re.doc_wise, lines, re.UNICODE)):
                            name, sentence = doc.groups()
                            biographies[key].append(self._remove_anchor_tag(sentence))
                    else:
                        biographies[key] = []
                        sentences = list(map(str.strip, open(os.path.join(self.corpus_folder, f'{key}.txt')).readlines()))
                        for sentence in sentences:
                            if sentence:
                                biographies[key].append(self._clean_sentence(sentence))

                    n_docs = len(biographies[key])
                    if n_docs < least_docs_key[1]:
                        least_docs_key = (gender, n_docs)

        return biographies, least_docs_key

    def _get_balanced_corpus(self, sentences,  language,  max_sentences, seed=15):
        """
        Balances a dataset composed by :param sentences by randomly deleting samples from a given key until it is equal to :param max_sentences
        :param sentences: sentences in the data set
        :type sentences: dict[str, list]
        :param max_sentences: least number of documents in the whole data set
        :type
        """
        random.seed(seed)
        balanced_dataset = list()
        for gender in GENDERS:
            key = f'{language}_{gender}'
            length = len(sentences[key])
            n_delete = length - max_sentences
            if n_delete > 0:
                print(f'Number of docs for gender: {gender} in language: {language} -> {length}. Randomly deleting {n_delete} '
                      f'documents to have a uniform distribution between genders')
                sample = random.sample(sentences[key], max_sentences)  # For reproducibility purpose
                balanced_dataset.extend(sample)
            else:
                balanced_dataset.extend(sentences[key])

        return balanced_dataset

    def save_sentences(self, format='xml'):
        """
        Saves a list of a list of words for the specified file names constructed by :param corpus_folder and :param languages
        Each sublist is a sentence.
        :return:
        """
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

    def process_file(self, filename):
        sentences_by_person = defaultdict(list)
        for line in open(filename):
            line = line.strip().lower()
            sentence, name = parse_sentence(line)
            if sentence:
                cleaned_sentence = self._clean_sentence(sentence)
                sentences_by_person[name].append(cleaned_sentence)

        return sentences_by_person

    def load_biographies_corpus(self, language):
        """

        :param language:
        :return:
        """
        fn = os.path.join(self.corpus_folder, f'biographies.corpus.tc.{language}')
        return open(fn).readlines()

    def generate_biographies_corpus(self, language):
        """

        :param language:
        :return:
        """
        merged_genders = list()
        for gender in self.genders:
            sentences = list(map(str.strip, open(os.path.join(self.corpus_folder, f'{language}_{gender}.txt')).readlines()))
            sentences_without_name = list()
            for sentence in sentences:
                if sentence:
                    clean_sent = self._clean_sentence(sentence)
                    sentences_without_name.append(clean_sent)

            merged_genders.extend(sentences_without_name)

        return merged_genders

    def load_balanced_corpus(self, language):
        """

        :param language:
        :return:
        """
        fn = os.path.join(self.corpus_folder, f'balanced.corpus.tc.{language}')
        return open(fn).readlines()

    def generate_balanced_corpus(self):
        """
        Generates a balanced dataset between the :param languages and :param genders key sets.
        """
        # Get docs to quantify how many he/she instances exist in each language.
        docs, (least_docs_key, least_docs_value) = self._get_biographies_corpus(format='txt')
        print(f'Key with least documents ({least_docs_value}): {least_docs_key}')

        # Balance dataset based on least_docs_val.
        for language in self.languages:
            balanced_dataset = self._get_balanced_corpus(docs, language, max_sentences=least_docs_value)
            with open(os.path.join(self.save_dir, f'balanced.corpus.tc.{language}'), 'w+') as f:
                f.writelines("\n".join(balanced_dataset))

    def load_europarl_corpus(self, language):
        """

        :param language:
        :return:
        """
        fn = os.path.join(self.corpus_folder, f'EuroParl.corpus.tc.{language}')
        return open(fn).readlines()
