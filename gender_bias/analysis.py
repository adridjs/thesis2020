import re
import logging
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, LabelSet, PanTool, \
    BoxZoomTool, ResetTool
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import random

from gender_bias.embeddings import Embeddings
from gebiotoolkit.storage_modules.storage_modules import find_pronouns


class Analysis:
    def __init__(self, language, corpus='balanced', filter_unbiased=False):
        self.corpus = corpus
        self.filter_unbiased = filter_unbiased
        self.embeddings = Embeddings(language, dataset=corpus)
        self.embeddings.load()
        self.as_dict = self.embeddings.as_dict()
        self.definitional_pairs = list(map(str.split, map(str.strip, open(f'data/{language}_definitional_pairs.txt').readlines())))
        self.professions = list(map(str.strip, open(f'data/{language}_professions.txt').readlines()))
        # self.stereo_male_words = list(map(str.strip, open(f'data/{language}_male_words.txt')))
        # self.stereo_female_words = list(map(str.strip, open(f'data/{language}_female_words.txt')))
        self.profession_embeddings = {token: emb for token, emb in self.as_dict.items() if token in self.professions}
        logging.basicConfig(filename=self.corpus + '.log',
                            filemode='w',
                            level='DEBUG')
        # logging.info(f'Vocabulary size: {len(self.embeddings.kv.vectors)}')
        # logging.info(f'Number of professions: {len(self.professions)}')
        # logging.info(f'Number of professions in vocabulary: {len(self.profession_embeddings)}')

    def plot_gendered_vectors(self, pair, data):
        """

        :param gendered_vectors:
        :param pair:
        :param threshold:
        :rtype
        """
        p = self._build_plot(pair, data)
        show(p)

    @staticmethod
    def _build_plot(pair, data, save=False):
        profession, similarity, gender_diff, gender = \
            list(zip(*[(prof, values['cos_similarity'], values['gender_score'], values['gender'])
                       for prof, values in data.items()]))
        source = ColumnDataSource(data=dict(similarity=similarity, difference=gender_diff, words=profession))
        p = figure(title=f"Words with gender diff != 0 in {pair} projection",
                   title_location="below",
                   x_range=[-1, 1],
                   y_range=[0, 1],
                   toolbar_location="above",
                   tools=[HoverTool(tooltips=[('Word', '@words'),
                                              ('Difference', '@difference'),
                                              ('Similarity', '@similarity')]),
                                PanTool(), BoxZoomTool(), ResetTool()])

        p.xaxis.axis_label = 'Cosine Similarity'
        p.yaxis.axis_label = 'Gender Score'
        p.circle(x='similarity', y='difference', size=8, source=source, line_color="black", fill_alpha=0.8)  # , color='gender'
        labels = LabelSet(x="similarity", y="difference", text="words", y_offset=8,
                          text_font_size="8pt", text_color="#555555",
                          source=source, text_align='center')
        logging.info(f'Pearson Correlation {pearsonr(similarity, gender_diff)[0]}')
        p.add_layout(labels)
        if save:
            output_file(f'{pair}')

        return p

    def analogy(self, a, b, c):
        """
        a is to b as c is to ?
        :return: most similar word to the analogy
        """
        result = self.embeddings.kv.most_similar_cosmul(positive=[a, c], negative=[b])
        logging.info("{}: {:.4f}".format(*result[0]))
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

    def _get_gender(self, prof):
        if prof in self.stereo_male_words:
            return 'male'
        elif prof in self.stereo_female_words:
            return 'female'
        else:
            return

    def _get_embedding_neighbors_info(self, gender_vector, n_neighbors=10):
        data = dict()
        for prof in self.profession_embeddings:
            neighbors = self.n_neighbors(prof, n_neighbors)
            gender = self._get_gender(prof)
            n_male = [neigh for neigh in neighbors if neigh[0] in self.stereo_male_words]
            n_female = [neigh for neigh in neighbors if neigh[0] in self.stereo_female_words]
            cos_similarity = np.dot(self.as_dict[prof], gender_vector)
            data.update({prof: dict(neighbors=neighbors,
                                                      n_male=n_male,
                                                      n_female=n_female,
                                                      cos_similarity=cos_similarity,
                                                      gender=gender,
                                                      gender_score=len(n_male)/n_neighbors)})

        return data

    def plot_gendered_vectors_by_pairs(self, n_neighbors=10, words_to_plot=10):
        """

        :param neighbors:
        :param words_to_plot:
        :return:
        """
        # definitional pairs
        # Based on arxiv:1903.03862v2
        logging.info('\n'*5 + f'Corpus: {self.corpus}')
        logging.info(f'Number of professions in vocabulary: {len(self.profession_embeddings)}')
        for gender_word_pair, gender_vector in self._get_gender_base().items():
            eni = self._get_embedding_neighbors_info(gender_vector,
                                                     n_neighbors=n_neighbors)
            masc_biased, fem_biased = \
                self._get_masculine_feminine_biased_words(eni,
                                                          words_to_plot=words_to_plot)
            logging.info(f'Masculine-biased: \t\n' + '\n'.join(word + '\n\t' +
                                                               '\n\t'.join(f'{word} {dist}'
                                                                           for word, dist
                                                                           in neigh_info['neighbors'][:5])
                                                               for word, neigh_info in masc_biased[:5]))
            logging.info(f'Feminine-biased: \t\n' +
                  '\n'.join(word + '\n\t' + '\n\t'.join(f'{word} {dist}' for word, dist in neigh_info['neighbors'][:5])
                            for word, neigh_info in fem_biased[:5]))
            if self.filter_unbiased:
                sorted_cosine = self.filter(gender_vector, n_neighbors=n_neighbors,
                                            words_to_plot=words_to_plot)

                data = dict(sorted_cosine[:words_to_plot]) if words_to_plot else dict(sorted_cosine)
                self.plot_gendered_vectors(gender_word_pair, data)
            else:
                self.plot_gendered_vectors(gender_word_pair, eni)

    def print_gender_stats(self, sentences, corpus):
        professions = Counter()
        raw_spaced = ' '.join(sentences)
        prof_genders = defaultdict(list)
        for profession in self.professions:
            professions.update({profession:
                                    len(re.findall(f'\s{profession}\s', raw_spaced))})
            gender = self._get_gender(profession)
            if gender:
                prof_genders[gender].append(profession)

        logging.info('\n' * 2 + f'{self.corpus}')
        logging.info(f'Most (10) common professions: '
                     f'{professions.most_common(10)}')
        logging.info(f'Professions by gender: \n\t'
                     f'male ({len(prof_genders["male"])}: {prof_genders["male"]} \n'
                     f'female ({len(prof_genders["female"])}: {prof_genders["female"]}')

        labels, values = zip(*sorted(professions.items(), key=lambda kv: kv[1],
                                     reverse=True)[:50])
        indexes = np.arange(len(labels))
        bar_width = 0.5
        plt.bar(indexes, values)
        # add kmeans_labels
        plt.margins(0.2)
        plt.xticks(indexes + bar_width, labels, rotation='vertical')
        plt.legend(corpus)
        plt.show()

    def _get_gender_base(self):
        gender_base = dict()
        for masc, fem in self.definitional_pairs:
            # gender vector is defined as the masculine vector minus the feminine vector
            gender_vector = self.as_dict[masc] - self.as_dict[fem]
            gender_base[(masc, fem)] = gender_vector
        return gender_base

    @staticmethod
    def _get_masculine_feminine_biased_words(neighbors_info,
                                             words_to_plot=None):

        masc_biased = list(sorted(neighbors_info.items(), key=lambda kv: len(kv[1]['n_male']),
                                  reverse=True))
        fem_biased = list(sorted(neighbors_info.items(), key=lambda kv: len(kv[1]['n_female']),
                                 reverse=True))
        masc_biased = \
            masc_biased[:words_to_plot] if words_to_plot else masc_biased
        fem_biased = \
            fem_biased[:words_to_plot] if words_to_plot else fem_biased

        return masc_biased, fem_biased

    def compute_direct_bias(self, gender_direction):
        prof_embs = [self.as_dict.get(profession) for profession in self.professions]
        prof_embs_filtered = list(map(lambda v: v.reshape(1, -1), filter(lambda x: x is not None, prof_embs)))
        cos_sims = [abs(cosine_similarity(prof_emb, gender_direction)) for prof_emb in prof_embs_filtered]
        print(f'Direct Bias ({self.corpus}: {sum(cos_sims)/len(prof_embs_filtered)}')
        print(f'N: {len(prof_embs_filtered)}')

    def plot_pca(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(20, 10))
        plt.subplots_adjust(wspace=0.5)
        ax1.xlabel = 'PCA 9 components'
        ax1.ylabel = 'Variance Ratio'
        ax2.xlabel = 'Random Dimensions'
        ax2.ylabel = 'Variance_Ratio'
        ax1.tick_params(labelsize=20)
        ax2.tick_params(labelsize=20)
        gender_base = list(self._get_gender_base().values())
        pca = PCA()
        model = pca.fit(gender_base)
        gender_direction = model.components_[0].reshape(1, -1)
        self.compute_direct_bias(gender_direction)
        n_components = len(model.explained_variance_ratio_)
        ax1.bar(range(1, n_components+1),
                model.explained_variance_ratio_,
                color='b')
        ax1.set_xticks(range(1, n_components))
        random_base = np.random.rand(9, 128)
        model = pca.fit(random_base)
        n_components = len(model.explained_variance_ratio_)
        ax2.bar(range(1, n_components+1),
                model.explained_variance_ratio_,
                color='b')
        ax2.set_xticks(range(1, n_components))
        ax2.tick_params(left=True, labelleft=True)
        plt.savefig(f'plots/PCA comparison-{self.corpus}.pdf', format='pdf')
        plt.show()

    def filter(self, gender_vector, n_neighbors=10, words_to_plot=None):
        eni = self._get_embedding_neighbors_info(gender_vector,
                                                 n_neighbors=n_neighbors)
        total = len(eni)
        eni_filtered = list(
            filter(lambda kv: kv[1]['gender_score'] != 0, eni.items()))
        logging.info(f'Number of professions with gender score != 0 in '
                     f'vocabulary:  {len(eni_filtered), len(eni_filtered) / total}')
        sorted_cosine = list(sorted(eni.items(),
                                    key=lambda keyval: abs(keyval[1]['cos_similarity']),
                                    reverse=True))
        sorted_cosine = sorted_cosine[:words_to_plot] if words_to_plot else sorted_cosine
        logging.info('-' * 120)
        logging.info(f'Gendered words by cosine similarity: \n\t' +
                     '\n\t'.join(f'{word} {neigh_info["cos_similarity"]}'
                                 for word, neigh_info in sorted_cosine))
        return sorted_cosine
