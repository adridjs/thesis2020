from matplotlib import pyplot
from gender_bias.clustering import Clustering


def main():
    n_clusters = 2
    min_count = 5
    window = 3
    for corpus in ['balanced', 'EuroParl']:
        clustering = Clustering(n_clusters=n_clusters, corpus=corpus, min_count=min_count, window=window)
        clustering.build_clusters_plot()
        clustering._write_log(filename=f'logs/clustering_{corpus}_nc:{n_clusters}_mn:{min_count}_w:{window}_filter:{True}.txt')
        pyplot.savefig(f'plots/clustering-n:{n_clusters}-mc:{min_count}-w:{window}-{corpus}-fw:{True}.pdf', format='pdf')


if __name__ == '__main__':
    main()
