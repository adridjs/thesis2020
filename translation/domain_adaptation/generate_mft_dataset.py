import argparse
import os
import random

from word2vec.utils.constants import LANGUAGES, GENDERS


class MFTDatasetGenerator:
    def __init__(self, corpus_folder, languages=None, genders=None, sampling_probs=None, prefixes=None):
        if sampling_probs and len(sampling_probs) != len(os.listdir(corpus_folder)):
            raise ValueError(f'Different length between sampling_probs {len(sampling_probs)} and number of files under {corpus_folder} ({len(os.listdir(corpus_folder))})')

        self.corpus_folder = corpus_folder
        self.probabilities = sampling_probs or []
        self.datasets = {}
        self.languages = languages or LANGUAGES
        self.genders = genders or GENDERS
        self.prefixes = prefixes or {'train': 'corpus.tc', 'test': 'test', 'dev': 'dev'}

    def load_original_corpus(self, language):
        fn = os.path.join(self.corpus_folder, f'corpus.clean.{language}')
        return open(fn).readlines()

    def load_biographies_corpus(self, language):
        """

        :param language:
        :return:
        """
        expected_fn = f'biographies.clean.{language}'
        if expected_fn not in os.listdir(self.corpus_folder):
            sentences = self.merge_gender_sentences(language)
        else:
            fn = os.path.join(self.corpus_folder, expected_fn)
            sentences = open(fn).readlines()

        return sentences

    def merge_gender_sentences(self, language):
        """

        :param language:
        :return:
        """
        merged = list()
        for gender in self.genders:
            sentences = open(os.path.join(self.corpus_folder, f'{language}_{gender}.filtered.txt')).readlines()
            merged.extend(sentences)

        return merged

    def split_dataset(self, dataset):
        """

        :param dataset:
        :return:
        """
        split = dict()
        n_samples = len(dataset)
        ratios = [0.8, 0.2]
        split['train'] = random.sample(dataset, int(n_samples*ratios[0]))
        remaining = [row for row in dataset if row not in split['train']]
        remaining_samples = len(remaining)
        split['test'], split['dev']= remaining[:int(remaining_samples/2)], remaining[int(remaining_samples/2):]

        return split

    def generate_mixed_dataset(self, seed=15):
        """

        :return:
        """
        random.seed(seed)
        for language in self.languages:
            biographies_corpus = self.load_biographies_corpus(language)
            original_corpus = self.load_original_corpus(language)
            original_corpus_subset = random.sample(original_corpus, len(biographies_corpus))
            # In order to mix samples from both datasets
            mixed_corpus = original_corpus_subset + biographies_corpus
            random.shuffle(mixed_corpus)
            # partition into train, test and dev
            split = self.split_dataset(mixed_corpus)
            for part_name, pref_name in self.prefixes.items():
                with open(os.path.join(self.corpus_folder, f'{pref_name}.{language}'), 'w+') as f:
                    f.writelines(split[part_name])


def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f,""--corpus_folder", dest='corpus_folder', help="paths to corpus data", default='data/')
    parser.add_argument("-l", "--languages", dest='languages', nargs='+', help="white-spaced languages you want to process the data on")

    return parser.parse_args()


def main():
    args = retrieve_args()

    corpus_folder = args.corpus_folder
    languages = args.languages

    dg = MFTDatasetGenerator(corpus_folder, languages=languages)
    dg.generate_mixed_dataset()


if __name__ == '__main__':
    main()
