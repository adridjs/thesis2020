import random
import re
import os
from collections import defaultdict, Counter

from gebiotoolkit.storage_modules.file_restructure import parse_sentence, save_xml
from utils.constants import GENDERS, LANGUAGES
from utils.regexp import RegExp


class DataDriver:
    """
    :param corpus_folder: Path to the folder where the aligned corpus_name is stored
    :type corpus_folder: str
    :param
    :param languages: Languages from which to extract corpus_name from :param corpus_folder:
    :type languages: set
    :param genders: Genders from which to extract corpus_name from :param corpus_folder:
    :type genders: set
    """
    def __init__(self, corpus_folder, save_dir=None, languages=None, genders=None):
        self.corpus_folder = corpus_folder
        self.save_dir = save_dir
        self.languages = languages or LANGUAGES
        self.genders = genders or GENDERS
        self.re = RegExp()

    @staticmethod
    def _remove_anchor_tag(txt):
        """
        Removes anchors from text.
        :param txt: Text with anchors
        :type txt: str
        :return: The text without anchors
        :rtype: str
        """
        pattern = r'<(a|/a).*?>'
        return re.sub(pattern, "", txt).strip()

    def clean_sentence(self, txt):
        """
        Cleans the sentence, removing the leading name and the hyphens.
        :param txt: Text to clean
        :type txt: str
        :return: Raw sentence
        :rtype: str
        """
        try:
            sentence = txt.strip().split(':')[1]
            result = self._remove_anchor_tag(sentence)
            return result
        except (ValueError, IndexError):
            print(f'Error when splitting sentence by ":" {txt}')

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

    def get_biographies_corpus(self, format='xml', filename=None):
        """
        Get the documents for each lang-gender key and return them together with the key that has the least documents
        :return: The documents as a dictionary and the key with least documents along with its value as a tuple
        :rtype: dict[str, list], tuple[str, int]
        """
        biographies = defaultdict(list)
        least_docs_key = (None, 10 ** 6)
        for lang in self.languages:
            for gender in self.genders:
                key = f'{lang}_{gender}'
                filename = filename or f'{self.corpus_folder}/{key}.{format}'
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
                                biographies[key].append(self.clean_sentence(sentence))

                    n_docs = len(biographies[key])
                    if n_docs < least_docs_key[1]:
                        least_docs_key = (gender, n_docs)

        return biographies, least_docs_key

    def get_balanced_corpus(self, sentences, language, max_sentences, seed=15):
        """
        Balances a corpus composed by :param sentences by randomly deleting samples from a given key until it is equal to :param max_sentences
        :param sentences: sentences in the corpus_name set
        :type sentences: dict[str, list]
        :param max_sentences: least number of documents in the whole corpus_name set
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
    
    def get_gebiocorpus_v2(self, language):
        c = Counter()
        gebiocorpus = []
        for gender in self.genders:
            filename = f'{gender}.1000.doc.{language}'
            with open(os.path.join(self.corpus_folder, filename), 'r') as f:
                lines = ' '.join(f.readlines())
                for n, doc in enumerate(re.finditer(self.re.doc_wise, lines, re.UNICODE)):
                    name, sentence = doc.groups()
                    c[name] += 1
                    gebiocorpus.append(self._remove_anchor_tag(sentence))
        return gebiocorpus, c

    def save_sentences(self, format='xml'):
        """
        Saves a list of a list of words for the specified file names constructed by :param corpus_folder and :param languages
        Each sublist is a sentence.
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
                cleaned_sentence = self.clean_sentence(sentence)
                sentences_by_person[name].append(cleaned_sentence)

        return sentences_by_person

    def _load_biographies_corpus(self, language):
        """

        :param language:
        :return:
        """
        fn = os.path.join(self.corpus_folder, f'biographies.corpus.tc.{language}')
        return open(fn).readlines()

    def _load_gender_balanced_corpus(self, language):
        """

        :param language:
        :return:
        """
        fn = os.path.join(self.corpus_folder, f'balanced.corpus.tc.{language}')
        return open(fn).readlines()

    def _load_europarl_corpus(self, language):
        """

        :param language:
        :return:
        """
        fn = os.path.join(self.corpus_folder, f'EuroParl.corpus.tc.{language}')
        return open(fn).readlines()

    def load_corpus(self, corpus_name, ratio=1.0):
        """
        Helper function to load corpus_name based on :param corpus_name name:. The ratio
        argument is only used when requesting to load the mixed corpus_name. By default,
        it appends all the EuroParl sentences to the balanced corpus_name. When different
        than 1, it appends the corresponding proportion of sentences from EuroParl.

        :param corpus_name: Name of the corpus_name to load
        :type corpus_name: str
        :param ratio: If set to different than 1, selects a portion of sentences to be
        added to the mixed corpus.
        :type ratio: float
        :return: The corpus_name keyed by language
        :rtype: dict[str, list[str]]
        """
        corpus_by_lang = dict()
        if corpus_name == 'mixed':
            for language in self.languages:
                orig_corpus = self._load_europarl_corpus(language)
                n_samples = int(len(orig_corpus)*ratio)
                gender_balanced = self._load_gender_balanced_corpus(language)
                corpus_by_lang[language] = gender_balanced + orig_corpus[:n_samples]
        elif corpus_name == 'europarl':
            for language in self.languages:
                corpus_by_lang[language] = self._load_europarl_corpus(language)
        elif corpus_name == 'biographies':
            for language in self.languages:
                corpus_by_lang[language] = self._load_biographies_corpus(language)
        elif corpus_name == 'balanced':
            for language in self.languages:
                corpus_by_lang[language] = self._load_gender_balanced_corpus(language)

        return corpus_by_lang
