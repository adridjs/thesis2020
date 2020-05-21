import argparse

from gender_bias.data_driver import DataDriver
from translation.MFTGenerator import MFTGenerator


def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--corpus_dir", dest='corpus_dir', default='data/',
                        help="paths to corpus data")
    parser.add_argument("-s", "--save_dir", dest='save_dir', default='partitions/',
                        help="directory where files will be saved")
    parser.add_argument("-l", "--languages", dest='languages', nargs='+',
                        help="white-spaced languages you want to process the data on")
    return parser.parse_args()


def main():
    args = retrieve_args()
    dd = DataDriver(args.corpus_dir, save_dir=args.save_dir)
    dg = MFTGenerator(args.corpus_dir)

    for n, language in enumerate(dd.languages):
        dg.datasets[language] = dd.load_europarl_corpus(language)

    # split into train, test and dev
    dg.split_dataset()


if __name__ == '__main__':
    main()
