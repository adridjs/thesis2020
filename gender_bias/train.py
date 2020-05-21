import argparse

from gender_bias.Word2Vec import Word2VecTrainer


def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cp", "--corpus-folder", dest='corpus_folder',
                        default='../translation/data',
                        help="path to the folder containing corpus")
    parser.add_argument("-l", "--language", dest='language',
                        default='en',
                        help="language to train the embeddings on")
    parser.add_argument("-s", "--save-binary", dest='save_binary',
                        default=False,
                        help="if set to true, the model will be saved as bytes.")
    parser.add_argument("-c", "--min-count", dest='min_count',
                        default=5,
                        help='minimum number of occurrences to add a word to the '
                             'vocabulary.')
    parser.add_argument("-w", "--window", dest='window',
                        default=3,
                        help='Number of words to right and left of the center word that are '
                             'considered.')
    parser.add_argument("-e", "--n-epochs", dest='n_epochs',
                        default=4,
                        help='minimum number of occurrences to add a word to the '
                              'vocabulary.')
    parser.add_argument("-es", "--embedding-size", dest='embedding_size',
                        default=128, help='minimum number of occurrences to add a word to '
                                          'the vocabulary.')
    parser.add_argument("-mn", "--model-name", dest='model_name',
                        help='filename of the word2vec model')
    parser.add_argument("--dataset", dest='dataset',
                        default='balanced',
                        help='name of the dataset used as input for the word embeddings '
                             'model: ["balanced", "biographies", "europarl"]')

    return parser.parse_args()


def main():
    args = retrieve_args()
    trainer = Word2VecTrainer(args.corpus_folder, args.language,
                              save_binary=args.save_binary,
                              min_count=int(args.min_count),
                              window=int(args.window),
                              n_epochs=int(args.n_epochs),
                              embedding_size=int(args.embedding_size),
                              model_name=args.model_name,
                              dataset=args.dataset)
    trainer.train()


if __name__ == '__main__':
    main()
