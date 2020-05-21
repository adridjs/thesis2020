import argparse
import os

from gender_bias.data_driver import DataDriver


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
    dd.generate_balanced_corpus()


if __name__ == '__main__':
    main()
