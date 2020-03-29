from gensim.models import KeyedVectors
import numpy as np


class Embeddings:
    def __init__(self, embeddings_filename, normalize=True):
        self.embeddings_filename = embeddings_filename
        self.normalize = normalize

    def __call__(self, *args, **kwargs):
        self.kv = KeyedVectors.load_word2vec_format(self.embeddings_filename, binary=kwargs.get('binary', False), encoding='utf-8')

    @property
    def as_dict(self):
        """
        Normalizes the L2 norm of vectors
        :param kv: KeyedVectors instansce with loaded vectors
        :return: Normalized embeddings as dictionary
        """
        return {word: emb / np.linalg.norm(emb) if self.normalize else emb for word, emb in zip(self.kv.index2word, self.kv.vectors)}