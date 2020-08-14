import os

import numpy as np
from gensim.models import KeyedVectors

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Embeddings:
    def __init__(self, language, min_count=5, window=3, embedding_size=128,
                 dataset='EuroParl', normalize=True):
        self.input_file = f'embeddings/{dataset}_l:{language}_mn:{min_count}_' \
                                   f'w:{window}_es:{embedding_size}.txt'
        self.normalize = normalize
        self.kv = None

    def load(self,  **kwargs):
        self.kv = KeyedVectors.load_word2vec_format(self.input_file,
                                                    binary=kwargs.get('binary', False),
                                                    encoding='utf-8')

    def as_dict(self):
        """
        Normalizes the L2 norm of vectors
        :param kv: KeyedVectors instansce with loaded vectors
        :return: Normalized embeddings as dictionary
        """
        if not self.kv:
            self.load()
        return {word: emb / np.linalg.norm(emb) if self.normalize else emb for
                word, emb in zip(self.kv.index2word, self.kv.vectors)}

    def save_tf(self):
        embeddings_vectors = np.stack(list(self.as_dict().values()), axis=0)
        # shape [n_words, embedding_size]
        emb = tf.Variable(embeddings_vectors, name='word_embeddings')
        words = '\n'.join(list(self.as_dict().keys()))
        with open(os.path.join('tensorboard', 'metadata.tsv'), 'w') as f:
            f.write(words)

        # Add an op to initialize the variable.
        init_op = tf.global_variables_initializer()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Later, launch the model, initialize the variables and save the
        # variables to disk.
        with tf.Session() as sess:
            sess.run(init_op)

            # Save the variables to disk.
            save_path = saver.save(sess, "tensorboard/model.ckpt")
            print(f"Model saved in path: {save_path}")


if __name__ == '__main__':
    e = Embeddings('en')
    e.load()
    e.save_tf()
