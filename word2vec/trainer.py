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
        self.model_filename = kwargs.get('model_filename', f'{self.language}_word2vec_{self.min_count}')
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

    def train(self):
        self.data_driver.get_balanced_dataset()
        joint_genders_dataset = [sentence for key_based in self.data_driver.balanced_dataset.values() for sentence in key_based]
        model = Word2Vec(min_count=self.settings.min_count, size=self.n_dim, window=5)
        model.build_vocab(joint_genders_dataset)
        # train model
        model.train(joint_genders_dataset, total_examples=model.corpus_count, epochs=model.iter)
        # save vectors
        model.wv.save_word2vec_format(f'{self.settings.model_filename}.txt', binary=False)

        if self.settings.save_binary:
            model.save(f'{self.settings.model_filename}.bin')


def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", dest='language', help="language to train the embeddings on", default='en')
    parser.add_argument("-i,""--input_folder", dest='input_folder', help="path to the folder containing the generated xmls",
                        default='biographies')
    parser.add_argument("-s", "--save_binary", dest='save_binary', help="if set to true, the model will be saved as bytes.", default=False)
    parser.add_argument("-c", "--min_count", dest='min_count', help='minimum number of occurrences to add a word to the vocabulary.', default=5)

    return parser.parse_args()


def main():
    args = retrieve_args()

    input_folder = args.input_folder
    language = args.language
    save_binary = args.save_binary
    min_count = args.min_count

    trainer = Word2VecTrainer(language=language, input_folder=input_folder, save_binary=save_binary, min_count=min_count)
    trainer.train()


if __name__ == '__main__':
    main()
