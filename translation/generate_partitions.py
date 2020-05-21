import argparse

from translation.CorpusGenerator import CorpusGenerator


def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--corpus_dir", dest='corpus_dir',
                        default='data/',
                        help="paths to corpus data")
    parser.add_argument("-s", "--save_dir", dest='save_dir',
                        default='partitions/',
                        help="directory where files will be saved")
    parser.add_argument("-l", "--languages", dest='languages',
                        nargs='+',
                        help="white-spaced languages you want to process the data on")
    parser.add_argument("--corpus", dest='corpus',
                        default='balanced',
                        help="corpus name to load")
    return parser.parse_args()


def main():
    args = retrieve_args()
    dg = CorpusGenerator(args.corpus_dir,
                         save_dir=args.save_dir,
                         languages=args.languages)
    dg.generate_partitions(args.corpus)


if __name__ == '__main__':
    main()
