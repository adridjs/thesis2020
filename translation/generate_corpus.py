import argparse

from translation.CorpusGenerator import CorpusGenerator


def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f,""--corpus_folder", dest='corpus_folder', help="paths to corpus_name corpus_name",
                        default='../translation/corpus/')
    parser.add_argument("-s,""--save_dir", dest='save_dir', help="paths where the generated corpus will be saved",
                        default='../translation/corpus')
    parser.add_argument("-l", "--languages", dest='languages', nargs='+', help="white-spaced languages you want to process the corpus_name on")
    parser.add_argument("--corpus_name", dest='corpus_name',
                        default='mixed',
                        help="corpus_name name to load")
    return parser.parse_args()


def main():
    args = retrieve_args()
    dg = CorpusGenerator(args.corpus_folder, args.save_dir, languages=args.languages)
    # dg.generate_corpus(args.corpus_name)
    dg.generate_corpus('balanced')


if __name__ == '__main__':
    main()
