import numpy as np
from bokeh.plotting import figure, show, output_file

from word2vec.embeddings import Embeddings


class Analysis:
    def __init__(self, language):
        self.embeddings = Embeddings(f'word2vec/{language}_word2vec_5.txt')
        self.embeddings = self.embeddings()

    @staticmethod
    def _plot_gendered_vectors(pair, gendered_vectors, threshold=None):
        """

        :param gendered_vectors:
        :param pair:
        :param threshold:
        :rtype
        """
        output_file("line.html")
        if threshold:
            words, values = zip(*list(filter(lambda x: abs(x[1]) > threshold, gendered_vectors)))
        else:
            words, values = zip(*gendered_vectors)

        p = figure(title=f'{pair}', x_range=words, y_range=(-1, 1))
        p.xaxis.major_label_orientation = np.pi / 4
        p.vbar(x=words, bottom=0, top=values, width=0.2)

        show(p)

    def analogy(self, a, b, c):
        """
        a is to b as c is to ?
        >>> self.embeddings.most_similar_cosmul(positive=['Man', 'Woman'], negative=['King'])
        >>> 'Queen'
        :return: most similar word to the analogy
        """
        result = self.embeddings.most_similar_cosmul(positive=[a, c], negative=[b])
        print("{}: {:.4f}".format(*result[0]))
        return result

    def n_neighbors(self, word, n=10):
        """
        Gets the top-n most similar neighbors in the embedding space by cosine similarity
        :param word: Word to search its neighbors
        :type word: str
        :param n: number of neighbors to search for
        :type n: int
        :rtype: list
        """
        return self.embeddings.most_similar(positive=[word], topn=n)

    def plot_gendered_vectors_by_pairs(self, language):
        professions_filename = f'word2vec/{language}_professions.txt'
        print(f'Vocabulary size: {len(self.embeddings.vectors)}')

        # definitional pairs
        pairs = [('he', 'she'), ('father', 'mother'), ('his', 'her'), ('man', 'woman'), ('boy', 'girl')]

        # Based on arxiv:1903.03862v2
        for masc, fem in pairs:
            gender_vector = self.embeddings[masc] - self.embeddings[fem]
            # cos(u,v) =u·v / ‖u‖‖v‖
            # cos(w1, w2) = w1·w2 if embeddings are normalized
            professions = list(map(str.strip, open(professions_filename).readlines()))
            profession_embeddings = {token: emb for token, emb in self.embeddings.items() if token in professions}
            print(f'Number of professions in vocabulary: {len(profession_embeddings)}')
            cos_similarity = [(word, np.dot(self.embeddings[word], gender_vector)) for word in profession_embeddings]

            gendered_words = sorted(cos_similarity, key=lambda x: x[1])
            for th in [0.05, 0.1, 0.15, 0.2]:
                try:
                    ana._plot_gendered_vectors((masc, fem), gendered_words, threshold=th)
                except ValueError:
                    pass


if __name__ == '__main__':
    lang = 'en'
    ana = Analysis(lang)
    ana.plot_gendered_vectors_by_pairs(language=lang)
