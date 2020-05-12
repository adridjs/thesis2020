import argparse

from gensim.models import Word2Vec

from data_driver import DataDriver


class Word2VecSettings:
    """
    Settings for the Word2VecTrainer class.
    :param languages: set of languages to get the embeddings from.
    """
    def __init__(self, **kwargs):
        self.save_binary = kwargs.get('save_binary', False)
        self.language = kwargs.get('language')
        self.min_count = kwargs.get('min_count', 5)
        self.model_filename = kwargs.get('model_name', f'{self.language}_word2vec_{self.min_count}')
        self.input_folder = kwargs.get('input_folder', 'biographies/')


class Word2VecTrainer:
    """
    Train a gensim Word2Vec model from scratch by passing  a set of :param languages.
    :param languages:
    :param n_dim:
    :param
    """
    def __init__(self, n_dim=512, epochs=20, **kwargs):
        self.n_dim = n_dim
        self.epochs = epochs
        self.settings = Word2VecSettings(**kwargs)
        self.data_driver = DataDriver(self.settings.input_folder, {self.settings.language})

    def train(self, language, biographies_dataset=None):

        if biographies_dataset:
            self.data_driver.get_balanced_dataset()
            sentences = [sentence for key_based in self.data_driver.balanced_dataset.values() for sentence in key_based if language in key_based]
        else:
            sentences = open(f'../translation/domain_adaptation/data/corpus.clean.{language}').readlines()

        model = Word2Vec(min_count=self.settings.min_count, size=self.n_dim, window=5)
        model.build_vocab(sentences)
        # train model
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        # save vectors
        model.wv.save_word2vec_format(f'{self.settings.model_filename}.txt', binary=False)

        if self.settings.save_binary:
            model.save(f'{self.settings.model_filename}.bin')


def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", dest='language',
                        help="language to train the embeddings on", required=True)
    parser.add_argument("-i", "--input_folder", dest='input_folder', default='biographies',
                        help="path to the folder containing the generated xmls")
    parser.add_argument("-s", "--save_binary", dest='save_binary', default=False,
                        help="if set to true, the model will be saved as bytes.",)
    parser.add_argument("-c", "--min_count", dest='min_count', default=5, metavar='int',
                        help='minimum number of occurrences to add a word to the vocabulary.')
    parser.add_argument("-m", "--model_name", dest='model_name',
                        help='name to save the model with')
    return parser.parse_args()


def main():
    args = retrieve_args()
    trainer = Word2VecTrainer(language=args.language,
                                                     input_folder=args.input_folder,
                                                     save_binary=args.save_binary,
                                                     min_count=int(args.min_count),
                                                     model_name=args.model_name)
    trainer.train('en')


if __name__ == '__main__':
    main()
