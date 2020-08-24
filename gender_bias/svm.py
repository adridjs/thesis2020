import random

from gender_bias.clustering import Clustering

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, corpus='balanced'):
        self.clustering = Clustering(corpus=corpus)
        self.corpus = corpus
        self.train = {}
        self.test = {}
        self.train_labels = []
        self.train_values = []
        self.test_labels = []
        self.test_values = []
        self.model = svm.SVC(kernel='linear')

    def score(self, train=None, test=None):
        self.clustering.select_values(filter_words=True)
        self._split(train=train, test=test)
        self.prepare_data()
        self.model.fit(self.train_values, self.train_labels)
        accuracy = self.model.score(self.test_values, self.test_labels)
        return accuracy

    def _split(self, train=None, test=None):
        word_value_label = list(zip(self.clustering.words, self.clustering.values, self.clustering.labels))
        if self.corpus == 'balanced':
            train_samples = random.sample(word_value_label, int(len(self.clustering.words)*0.2))
            self.train = {word: (value, label) for word, value, label in train_samples}
            self.test = {word: (value, label) for word, value, label in word_value_label if word not in self.train}
        elif self.corpus == 'EuroParl':
            if not train or not test:
                raise ValueError('You need to provide the same samples than in "balanced"')
            aux_train = {word: (value, label) for word, value, label in word_value_label
                         if word in train}
            aux_test = {word: (value, label) for word, value, label in word_value_label
                         if word in test}
            self.train = aux_train
            self.test = aux_test

    def plot_discriminant(self, title):
        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        X = np.array(list(zip(*self.test_values))).reshape(61, 128)
        xx, yy = self.make_meshgrid(X)
        Z = self.model.predict(X)

        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=self.test_labels, cmap=plt.cm.coolwarm)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(title)

    def make_meshgrid(self, X, h=.02):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, y
        y : ndarray
        """
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(self, ax, clf, xx, yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    def plot_disc(self):
        X_train = np.array(list(zip(*self.train_values))).reshape(-1, 128)
        self.model.fit(X_train, self.train_labels)

        X_test = np.array(list(zip(*self.test_values))).reshape(-1, 128)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=self.test_labels, s=30, cmap=plt.cm.Paired)

        # plot the decision function
        ax = plt.gca()
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        #
        # # create grid to evaluate model
        # xx = np.linspace(xlim[0], xlim[1], 30)
        # yy = np.linspace(ylim[0], ylim[1], 30)
        # YY, XX = np.meshgrid(yy, xx)
        # xy = np.vstack([XX.ravel(), YY.ravel()] + [np.repeat(0, XX.ravel().size).T for _ in range(126)]).T
        # Z = self.model.decision_function(xy).reshape(XX.shape)
        # # plot decision boundary and margins
        # ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
        #            linestyles=['--', '-', '--'])
        # plot support vectors
        # ax.scatter(self.model.support_vectors_[:, 0], self.model.support_vectors_[:, 1], s=100,
        #            linewidth=1, facecolors='none', edgecolors='k')
        plt.savefig(f'svm-{self.corpus}.pdf', format='pdf')
        plt.show()


    def prepare_data(self):
        self.train_values, self.train_labels = list(zip(*self.train.values()))
        self.test_values, self.test_labels = list(zip(*self.test.values()))