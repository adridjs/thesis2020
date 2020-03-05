from gensim.models import Word2Vec

from word2vec.data_driver import DataDriver


class Word2VecSettings:
    """
    Settings for the Word2VecTrainer class.
    :param languages: set of languages to get the embeddings from.
    """
    def __init__(self, languages, kwargs):
        self.save_binary = kwargs.get('save_binary', False)
        self.min_count = kwargs.get('min_count', 5)
        self.model_filename = kwargs.get('model_filename', f'biographies_word2vec_{self.min_count}')
        self.input_folder = kwargs.get('input_folder', 'biographies/')
        self.data_loader = DataDriver(self.input_folder, languages)


class Word2VecTrainer:
    """
    Train a gensim Word2Vec model from scratch by passing  a set of :param languages.
    :param languages:
    :param n_dim:
    :param
    """
    def __init__(self, languages, n_dim=512, epochs=20, **kwargs):
        self.n_dim = n_dim
        self.epochs = epochs
        self.settings = Word2VecSettings(languages, kwargs)

    def train(self):
        dataset = self.settings.data_loader.get_balanced_dataset()
        joint_genders_dataset = [sentence for gender_based in dataset.values() for sentence in gender_based]
        model = Word2Vec(sentences=joint_genders_dataset, size=self.n_dim, min_count=self.settings.min_count)

        # train model
        model.train(epochs=self.epochs)  # train word vectors

        # save vectors
        model.wv.save_word2vec_format(f'{self.settings.model_filename}.txt', binary=False)

        if self.settings.save_binary:
            model.save(f'{self.settings.model_filename}.bin')
