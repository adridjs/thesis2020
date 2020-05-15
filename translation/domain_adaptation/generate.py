import argparse

from gender_bias.data_driver import DataDriver
from translation.domain_adaptation.MFTGenerator import MFTGenerator


def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--corpus_dir", dest='corpus_dir', default='data/',
                        help="paths to corpus data")
    parser.add_argument("-s", "--save_dir", dest='save_dir', default='data/',
                        help="directory where files will be saved")
    parser.add_argument("-l", "--languages", dest='languages', nargs='+',
                        help="white-spaced languages you want to process the data on")
    parser.add_argument("--balanced", dest='balanced', action='store_true',
                        help="use balanced dataset if set")
    return parser.parse_args()


def main():
    args = retrieve_args()
    balanced = args.balanced
    dd = DataDriver(args.corpus_dir, save_dir=args.save_dir)
    dg = MFTGenerator(args.corpus_folder)

    for n, language in enumerate(dd.languages):
        orig_corpus = dd.load_europarl_corpus(language)
        if balanced:
            gender_balanced = dd.load_balanced_corpus(language)
            corpus = gender_balanced + orig_corpus
        else:
            bio_corpus = dd.load_biographies_corpus(language)
            corpus = bio_corpus + orig_corpus

        dg.datasets[language] = corpus

    sentence_pairs = tuple(dg.datasets.items())
    # split into train, test and dev
    dg.split_dataset(sentence_pairs)


if __name__ == '__main__':
    main()
