from gensim.models import Word2Vec

from gender_bias.DataDriver import DataDriver


class Word2VecSettings:
    """
    Settings for the Word2VecTrainer class.
    :param languages: set of languages to get the embeddings from.
    """
    def __init__(self, **kwargs):
        self.save_binary = kwargs.get('save_binary', False)
        self.language = kwargs.get('language', 'en')
        self.min_count = kwargs.get('min_count', 5)
        self.window = kwargs.get('window', 3)
        self.epochs = kwargs.get('epochs', 10)
        self.embedding_size = kwargs.get('embedding_size', 128)
        self.dataset = kwargs.get('dataset', 'EuroParl')
        model_name = kwargs.get('model_name')
        self.model_name = model_name or \
                        f'word2vec_l:{self.language}_'\
                        f'mn:{self.min_count}_'\
                        f'w:{self.window}_' \
                        f'es:{self.embedding_size}'


class Word2VecTrainer:
    """
    Train a gensim Word2Vec model from scratch by passing  a set of :param languages.
    :param languages:
    :param n_dim:
    :param
    """
    def __init__(self, corpus_folder, language, **kwargs):

        self.settings = Word2VecSettings(**kwargs)
        self.dd = DataDriver(corpus_folder, language)

    def train(self):
        sentences = \
            open(f'{self.dd.corpus_folder}/'f'{self.settings.dataset}.corpus.tc.{self.settings.language}')\
            .readlines()
        sentences_tokenized = [sentence.split() for sentence in sentences]
        model = Word2Vec(min_count=self.settings.min_count,
                         size=self.settings.embedding_size,
                         window=self.settings.window)
        model.build_vocab(sentences_tokenized)
        # train model
        model.train(sentences_tokenized,
                    total_examples=model.corpus_count,
                    epochs=self.settings.epochs)
        # save vectors
        model.wv.\
            save_word2vec_format(f'{self.settings.model_name}.txt', binary=False)

        if self.settings.save_binary:
            model.save(f'{self.settings.model_name}.bin')
