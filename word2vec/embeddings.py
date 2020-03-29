from gensim.models import KeyedVectors
import numpy as np


class Embeddings:
    def __init__(self, embeddings_filename):
        self.embeddings_filename = embeddings_filename

    def __call__(self, *args, **kwargs):
        kv = KeyedVectors.load_word2vec_format(self.embeddings_filename, kwargs.get('binary', False))
        return kv

    def normalize(self, kv):
        """
        Normalizes the L2 norm of vectors
        :param kv: KeyedVectors instansce with loaded vectors
        :return: Normalized embeddings as dictionary
        """
        return {word: emb / np.linalg.norm(emb) for word, emb in zip(kv.index2word, kv.vectors)}
