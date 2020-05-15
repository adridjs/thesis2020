import os
import random

from gender_bias.data_driver import DataDriver
from gender_bias.utils.constants import LANGUAGES, GENDERS


class MFTGenerator:
    """

    """
    def __init__(self, corpus_folder, save_dir=None, train_ratio=None, languages=None, genders=None, sampling_probs=None, prefixes=None):
        """

        :param corpus_folder:
        :param train_ratio:
        :param languages:
        :param genders:
        :param sampling_probs:
        :param prefixes:
        """
        if sampling_probs and len(sampling_probs) != len(os.listdir(corpus_folder)):
            raise ValueError(f'Different length between sampling_probs {len(sampling_probs)} and number of files under {corpus_folder} ({len(os.listdir(corpus_folder))})')

        self.corpus_folder = corpus_folder
        self.train_ratio = train_ratio or 0.8
        self.languages = languages or LANGUAGES
        self.genders = genders or GENDERS
        self.prefixes = prefixes or {'train': 'corpus.tc', 'test': 'test', 'dev': 'dev'}
        self.datasets = {}
        self.dd = DataDriver(corpus_folder, save_dir=save_dir or corpus_folder, languages=languages, genders=genders)

    def generate_indices(self, size):
        indices = set(range(size))
        train_indices = set(random.sample(indices, int(self.train_ratio * len(indices))))
        remaining_indices = indices.difference(train_indices)
        test_indices = set(random.sample(remaining_indices, int(0.5 * len(remaining_indices))))
        dev_indices = remaining_indices.difference(test_indices)

        return train_indices, test_indices, dev_indices

    def split_dataset(self, sentence_pairs):
        """

        :param sentence_pairs:  Tuple of language-sentence pairs. At the first level, the first index is the language, the second are the sentences.
        :type sentence_pairs: tuple of tuple[str, list]
        :return:
        """

        dataset_length = len(sentence_pairs[0][1])
        train, test, dev = self.generate_indices(dataset_length)
        for language, sentences in sentence_pairs:
            try:
                with open(os.path.join(self.corpus_folder, f'{self.prefixes["train"]}.{language}'), 'w+') as f:
                    x = [sentences[idx] for idx in train]
                    f.writelines(x)
                with open(os.path.join(self.corpus_folder, f'{self.prefixes["test"]}.{language}'), 'w+') as f:
                    x = [sentences[idx] for idx in test]
                    f.writelines(x)
                with open(os.path.join(self.corpus_folder, f'{self.prefixes["dev"]}.{language}'), 'w+') as f:
                    x = [sentences[idx] for idx in dev]
                    f.writelines(x)
            except IndexError:
                print(f'sdklfjdlkfg')

    def generate_mixed_dataset(self):
        """
        :return:
        """
        for n, language in enumerate(self.languages):
            bio_corpus = self.dd.generate_biographies_corpus(language)
            orig_corpus = self.dd.load_europarl_corpus(language)
            if n == 0:
                orig_idx, orig_subset = list(zip(*random.sample(list(enumerate(orig_corpus)), len(bio_corpus))))
            else:
                orig_subset = [orig_corpus[idx] for idx in orig_idx]

            self.datasets[language] = bio_corpus + list(orig_subset)

        sentence_pairs = tuple(self.datasets.items())
        # split into train, test and dev
        self.split_dataset(sentence_pairs)
