from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib import pyplot

from gender_bias.embeddings import Embeddings


class Clustering:
    def __init__(self, n_clusters=2, language='en', corpus='balanced',  min_count=5, window=3,
                 embedding_size=128):
        self.embeddings = Embeddings(language, dataset=corpus, min_count=min_count, window=window, embedding_size=embedding_size)
        self.embeddings.load()
        self.n_clusters = n_clusters
        self.as_dict = self.embeddings.as_dict()
        self.professions = list(map(str.strip, open(f'data/{language}_professions.txt').readlines()))
        self.stereo_male_words = set(map(str.strip, open(f'data/{language}_male_words.txt')))
        self.stereo_female_words = set(map(str.strip, open(f'data/{language}_female_words.txt')))
        self.word2label = {}
        self.kmeans_word2label = {}
        self.gendered_words = defaultdict(list)
        self.kmeans_labels = None
        self.clusters = None
        self.words = []
        self.labels = []

    def _get_gender(self, word):
        self.words.append(word)
        if word in self.stereo_male_words:
            self.labels.append(0)
            self.word2label[word] = 0
            self.gendered_words['male'].append(word)
            return 'male'
        elif word in self.stereo_female_words:
            self.labels.append(1)
            self.word2label[word] = 1
            self.gendered_words['female'].append(word)
            return 'female'

    def select_values(self, filter_words=False):
        if not filter_words:
            self.words, self.values = list(zip(*self.as_dict.items()))
        else:
            values_to_cluster = list()
            words = list()
            for word, emb in self.as_dict.items():
                if self._get_gender(word):
                    words.append(word)
                    values_to_cluster.append(emb)
            self.words = words
            self.values = values_to_cluster

    def t_sne(self):
        self.cluster()
        tsne = TSNE().fit_transform(self.values)
        return list(zip(*list(tsne)))

    def cluster(self, filter_words=True):
        self.select_values(filter_words=filter_words)
        kmeans = KMeans(n_clusters=self.n_clusters).fit(self.values)
        self.kmeans_labels = kmeans.labels_
        self._build_cluster_dict()

    def _build_cluster_dict(self):
        clusters = defaultdict(list)
        for word, label in zip(self.as_dict.keys(), self.kmeans_labels):
            clusters[label].append(word)
            self.kmeans_word2label[word] = label
        self.clusters = clusters

    def build_clusters_plot(self, i):
        fig = pyplot.figure(i)
        ax = pyplot.axes()
        x, y = self.t_sne()
        label_wise_points = defaultdict(list)
        for x_point, y_point, label, word in zip(x, y, self.kmeans_labels, self.words):
            label_wise_points[label].append((x_point, y_point, word))
        for label, values in label_wise_points.items():
            x_points, y_points, words = list(zip(*values))
            ax.scatter(x_points, y_points, label=f'Cluster {label}')
            for i, (x, y, word) in enumerate(zip(x_points, y_points, words)):
                if i % 4:
                    ax.annotate(word, (x + 0.2, y + 0.2))
        pyplot.legend()
        return fig
