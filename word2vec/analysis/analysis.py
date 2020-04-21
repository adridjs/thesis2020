import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
from scipy.stats import pearsonr


from word2vec.embeddings import Embeddings


class Analysis:
    def __init__(self, language, filter_unbiased=False):
        self.filter_unbiased = filter_unbiased
        self.embeddings = Embeddings(f'../{language}_word2vec_5.txt')
        self.embeddings()
        print(f'Vocabulary size: {len(self.embeddings.kv.vectors)}')
        self.as_dict = self.embeddings.as_dict()
        self.definitional_pairs = list(map(str.split, map(str.strip, open(f'../data/{language}_definitional_pairs.txt').readlines())))
        self.professions = list(map(str.strip, open(f'../data/{language}_professions.txt').readlines()))
        print(f'Number of professions: {len(self.professions)}')
        self.profession_embeddings = {token: emb for token, emb in self.as_dict.items() if token in self.professions}
        print(f'Number of professions in vocabulary: {len(self.profession_embeddings)}')
        self.stereo_male_words = list(map(str.strip, open(f'../data/{language}_male_words.txt')))
        self.stereo_female_words = list(map(str.strip, open(f'../data/{language}_female_words.txt')))

    @staticmethod
    def _plot_gendered_vectors(pair, data, threshold=None, top_k=None):
        """

        :param gendered_vectors:
        :param pair:
        :param threshold:
        :rtype
        """
        colors = {'male': 'green', 'female': 'blue', 'x': 'black'}
        profession, similarity, gender_diff, gender = \
            list(zip(*[(prof, values['cos_similarity'], len(values['n_male']) - len(values['n_female']), values['gender'])
                                                    for prof, values in data.items()]))
        source = ColumnDataSource(data=dict(similarity=similarity, difference=gender_diff, words=profession))
        p = figure(tools=[HoverTool(tooltips=[('Word', '@words'),
                                              ('Difference', '@difference'),
                                              ('Similarity', '@similarity')])])
        p.circle(x='similarity', y='difference', size=8, source=source, line_color="black", fill_alpha=0.8, color='gender')
        labels = LabelSet(x="similarity", y="difference", text="words", y_offset=8,
                          text_font_size="8pt", text_color="#555555",
                          source=source, text_align='center')
        p.add_layout(labels)
        print(f'Pearson Correlation {pearsonr(similarity, gender_diff)[0]}')
        output_file(f'{pair}')
        show(p)

    def analogy(self, a, b, c):
        """
        a is to b as c is to ?
        :return: most similar word to the analogy
        """
        result = self.embeddings.kv.most_similar_cosmul(positive=[a, c], negative=[b])
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
        return self.embeddings.kv.most_similar(positive=[word], topn=n)

    def get_embedding_neighbors(self, gender_vector, n_neighbors=10):
        data = dict()
        for prof in self.profession_embeddings:
            neighbors = self.n_neighbors(prof, n_neighbors)
            if prof in self.stereo_male_words:
                gender = 'male'
            elif prof in self.stereo_female_words:
                gender = 'female'
            else:
                gender = None
            n_male = [neigh for neigh in neighbors if neigh[0] in self.stereo_male_words]
            n_female = [neigh for neigh in neighbors if neigh[0] in self.stereo_female_words]
            cos_similarity = np.dot(self.as_dict[prof], gender_vector)
            data.update({prof: dict(neighbors=neighbors, n_male=n_male, n_female=n_female, cos_similarity=cos_similarity, gender=gender)})

        return data

    def plot_gendered_vectors_by_pairs(self, n_neighbors=10, words_to_plot=10):
        """

        :param neighbors:
        :param words_to_plot:
        :return:
        """
        # definitional pairs
        # Based on arxiv:1903.03862v2
        for masc, fem in self.definitional_pairs:
            gender_vector = self.as_dict[masc] - self.as_dict[fem]
            # cos(u,v) =u·v / ‖u‖‖v‖
            # cos(w1, w2) = w1·w2 if embeddings are normalized
            print(f'Number of professions in vocabulary: {len(self.profession_embeddings)}')
            data = self.get_embedding_neighbors(gender_vector, n_neighbors=n_neighbors)
            if self.filter_unbiased:
                total = len(data)
                data = dict(list(filter(lambda kv: len(kv[1]['n_male']) - len(kv[1]['n_female']) != 0, data.items())))
                print(f'Number of professions with gender bias in vocabulary: {len(data), len(data)/total}')
            # masc_biased = sorted(data.items(), key=lambda kv: len(kv[1]['n_male']), reverse=True)
            # fem_biased = sorted(data.items(), key=lambda kv: len(kv[1]['n_female']), reverse=True)
            # gendered_words_cosine = sorted(data.items(), key=lambda keyval: keyval[1]['cos_similarity'], reverse=True)[words_to_plot]
            ana._plot_gendered_vectors((masc, fem), data, top_k=10)


if __name__ == '__main__':
    lang = 'en'
    ana = Analysis(lang, filter_unbiased=True)
    ana.plot_gendered_vectors_by_pairs(n_neighbors=10)
