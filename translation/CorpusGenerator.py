import os
import random

from gender_bias.data_driver import DataDriver


class CorpusGenerator:
    """
    Class used to generate the different corpus that are going to be analyzed.
    """
    def __init__(self, corpus_folder,
                 save_dir=None,
                 languages=None,
                 genders=None):
        """
        :param languages: Languages for which the corpus are going to be created
        :param genders: Genders for which the corpus are going to be created
        """

        self.supported_corpus = ['mixed', 'europarl', 'biographies', 'balanced', 'gebio']
        self.datasets = {}
        self.dd = DataDriver(corpus_folder,
                             save_dir=save_dir or corpus_folder,
                             languages=languages,
                             genders=genders)

    def _generate_biographies_corpus(self):
        """
        Generates Biographies Corpus
        """
        merged_genders = list()
        for language in self.dd.languages:
            for gender in self.dd.genders:
                sentences = list(map(str.strip,
                                     open(os.path.join(self.dd.corpus_folder,
                                                       f'{language}_{gender}.txt')).readlines()))
                sentences_without_name = list()
                for sentence in sentences:
                    if sentence:
                        clean_sent = self.dd.clean_sentence(sentence)
                        sentences_without_name.append(clean_sent)

                merged_genders.extend(sentences_without_name)

            with open(os.path.join(self.dd.save_dir, f'biographies.corpus.tc.{language}'), 'w+') as f:
                f.writelines("\n".join(merged_genders))

    def _generate_gender_balanced_corpus(self):
        """
        Generates GB corpus
        """
        # Get docs to quantify how many he/she instances exist in each language.
        docs, (least_docs_key, least_docs_value) = self.dd.get_biographies_corpus(format='txt')
        print(f'Key with least documents ({least_docs_value}): {least_docs_key}')

        # Balance corpus based on least_docs_val.
        for language in self.dd.languages:
            balanced_dataset = self.dd.get_balanced_corpus(docs, language, max_sentences=least_docs_value)
            with open(os.path.join(self.dd.save_dir, f'balanced.corpus.tc.{language}'), 'w+') as f:
                f.writelines("\n".join(balanced_dataset))

    def _generate_gebiocorpus_v2(self):
        """
        Generates Gebiocorpus_v2
        """
        counters = dict()
        for language in self.dd.languages:
            gebiocorpus, cnt = self.dd.get_gebiocorpus_v2(language)
            counters[language] = cnt
            with open(os.path.join(self.dd.save_dir, f'gebio.corpus.tc.{language}'), 'w+') as f:
                f.writelines("\n".join(gebiocorpus))

        for name, ct in counters['es'].items():
            if counters['en'][name] != ct:
                raise ValueError(name)

    def _generate_mixed_corpus(self, ratio=1.0):
        """
        Generates MFT Corpus between EuroParl and GB corpus
        :param ratio: If set to different than 1, selects a portion of sentences to be
        added to the mixed corpus.
        :type ratio: float
        """
        mixed_corpus = self.dd.load_corpus('mixed', ratio=ratio)
        for language in self.dd.languages:
            with open(os.path.join(self.dd.save_dir, f'mixed-{ratio}.corpus.tc.{language}'), 'w+') as f:
                f.writelines(mixed_corpus[language])

    def generate_corpus(self, corpus_name, ratio=1.0):
        """
        Writes to disk the corresponding corpus passed in :param corpus_name:
        :param corpus_name: Corpus to be created
        :type corpus_name: str
        :param ratio: If set to different than 1, selects a portion of sentences to be
        added to the mixed corpus.
        :type ratio: float
        """
        if corpus_name not in self.supported_corpus:
            raise ValueError(f'Supported corpus_name names are: {self.supported_corpus}')

        if corpus_name == 'biographies':
            self._generate_biographies_corpus()
        elif corpus_name == 'balanced':
            self._generate_gender_balanced_corpus()
        elif corpus_name == 'mixed':
            self._generate_mixed_corpus(ratio=ratio)
        elif corpus_name == 'gebio':
            self._generate_gebiocorpus_v2()
