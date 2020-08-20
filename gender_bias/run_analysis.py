from gender_bias.analysis import Analysis
from gender_bias.data_driver import DataDriver

if __name__ == '__main__':
    language = 'en'
    corpus_folder = '../translation/corpus/'
    dd = DataDriver(corpus_folder=corpus_folder, languages={'en'})
    for corpus in ['balanced', 'EuroParl']:
        analysis = Analysis(language, corpus=corpus)
        analysis.plot_pca()
        # sentences = list(map(str.strip, dd.load_corpus(corpus)[language]))
        # analysis.print_gender_stats(sentences, corpus)
        # analysis.plot_gendered_vectors_by_pairs(n_neighbors=10, words_to_plot=10)
