from matplotlib import pyplot
from gender_bias.clustering import Clustering


def main():
    n_clusters = 2
    min_count = 5
    window = 3
    corpus = 'balanced'
    clustering_balanced = Clustering(n_clusters=n_clusters, corpus=corpus, min_count=min_count, window=window)
    fig = clustering_balanced.build_clusters_plot()
    clustering_balanced._write_log(filename=f'logs/clustering_{corpus}_nc:{n_clusters}_mn:{min_count}_w:{window}_filter:{True}.txt')
    pyplot.savefig(f'plots/clustering-n:{n_clusters}-mc:{min_count}-w:{window}-{corpus}-fw:{True}.pdf', format='pdf')
    fig.title = f'{n_clusters}-{min_count}-{window}-{corpus}-{True}'
    corpus = 'EuroParl'
    clustering_EP = Clustering(n_clusters=n_clusters, corpus=corpus, min_count=min_count, window=window)
    fig = clustering_EP.build_clusters_plot()
    clustering_EP._write_log(filename=f'logs/clustering_{corpus}_nc:{n_clusters}_mn:{min_count}_w:{window}_filter:{True}.txt')
    pyplot.savefig(f'plots/clustering-n:{n_clusters}-mc:{min_count}-w:{window}-{corpus}-fw:{True}.pdf', format='pdf')
    fig.title = f'{n_clusters}-{min_count}-{window}-{corpus}-{True}'
    pyplot.show()


if __name__ == '__main__':
    main()