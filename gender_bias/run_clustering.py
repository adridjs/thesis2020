from matplotlib import pyplot
from gender_bias.clustering import Clustering


def main():
    corpus = 'EuroParl'
    # n_clusters_values = [2, 3, 4, 5]
    n_clusters_values = [2]
    # min_count_values = [2, 3, 4, 5]
    min_count_values = [5]
    # window_values = [2, 3]
    window_values = [3]

    i = 0
    for n_clusters in n_clusters_values:
        for min_count in min_count_values:
            for window in window_values:
                    with open(f'logs/clustering_{corpus}_nc:{n_clusters}_mn:{min_count}_w:{window}_filter:{True}.txt', 'w+') as f:
                        clustering = Clustering(n_clusters=n_clusters, corpus=corpus, min_count=min_count, window=window)
                        # for gender, words in clustering.gendered_words.items():
                        #     f.write(f'Gender: {gender}\n\t {words}\n')
                        # items = clustering.clusters.values()
                        # f.write(f'Number of stereotypically-gendered words in embeddings: {sum(len(itm) for itm in list(items))}\n')
                        # for label, words in clustering.clusters.items():
                        #     intersection_fem = set(words).intersection(clustering.gendered_words['female'])
                        #     intersection_masc = set(words).intersection(clustering.gendered_words['male'])
                        #     f.writelines(f'Cluster {label}: \n\t {words}\n')
                        #     f.write(f'\tFeminine Gendered words (%): {len(intersection_fem) / len(clustering.gendered_words["female"]) * 100}\n')
                        #     f.write(f'\tMasculine Gendered words (%): {len(intersection_masc) / len(clustering.gendered_words["male"]) * 100}\n')
                        fig = clustering.build_clusters_plot(i)
                        i += 1
                        pyplot.savefig(f'{n_clusters}-{min_count}-{window}-{True}.pdf', format='pdf')
                        fig.title = f'{n_clusters}-{min_count}-{window}-{True}'
    pyplot.show()


if __name__ == '__main__':
    main()