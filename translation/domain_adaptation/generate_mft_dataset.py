import argparse
import os
import random

from gender_bias.utils.constants import LANGUAGES, GENDERS


class MFTDatasetGenerator:
    """

    """
    def __init__(self, corpus_folder, train_ratio=None, languages=None, genders=None, sampling_probs=None, prefixes=None):
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

    def load_biographies_corpus(self, language):
        """

        :param language:
        :return:
        """
        fn = os.path.join(self.corpus_folder, f'biographies.corpus.tc.{language}')
        return open(fn).readlines()

    def load_balanced_corpus(self, language):
        """

        :param language:
        :return:
        """
        fn = os.path.join(self.corpus_folder, f'balanced.corpus.tc.{language}')
        return open(fn).readlines()

    def load_original_corpus(self, language):
        """

        :param language:
        :return:
        """
        fn = os.path.join(self.corpus_folder, f'corpus.clean.{language}')
        return open(fn).readlines()

    def merge_gender_sentences(self, language):
        """

        :param language:
        :return:
        """
        merged = list()
        for gender in self.genders:
            sentences = open(os.path.join(self.corpus_folder, f'{language}_{gender}.balanced.txt')).readlines()
            merged.extend(sentences)

        return merged

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
            bio_corpus = self.merge_gender_sentences(language)
            orig_corpus = self.load_original_corpus(language)
            if n == 0:
                orig_idx, orig_subset = list(zip(*random.sample(list(enumerate(orig_corpus)), len(bio_corpus))))
            else:
                orig_subset = [orig_corpus[idx] for idx in orig_idx]

            self.datasets[language] = bio_corpus + list(orig_subset)

        sentence_pairs = tuple(self.datasets.items())
        # split into train, test and dev
        self.split_dataset(sentence_pairs)

    def generate_biograpies_corpus(self):
        for language in self.languages:
            bio_corpus = self.merge_gender_sentences(language)
            with open(os.path.join(self.corpus_folder, f'balanced.corpus.tc.{language}'), 'w+') as f:
                f.writelines(bio_corpus)


def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f,""--corpus_folder", dest='corpus_folder', help="paths to corpus data", default='data/')
    parser.add_argument("-l", "--languages", dest='languages', nargs='+', help="white-spaced languages you want to process the data on")
    parser.add_argument("--repeat", dest='repeat_in_domain')

    return parser.parse_args()


def main():
    args = retrieve_args()
    dg = MFTDatasetGenerator(args.corpus_folder)
    # for language in dg.languages:
    #     dg.datasets[language] = dg.load_balanced_corpus(language)
    #
    # sentence_pairs = tuple(dg.datasets.items())
    #
    # # split into train, test and dev
    # dg.split_dataset(sentence_pairs)
    for n, language in enumerate(dg.languages):
        bio_corpus = dg.merge_gender_sentences(language)
        orig_corpus = dg.load_original_corpus(language)
        dg.datasets[language] = bio_corpus + orig_corpus

    sentence_pairs = tuple(dg.datasets.items())
    # split into train, test and dev
    dg.split_dataset(sentence_pairs)


if __name__ == '__main__':
    main()
