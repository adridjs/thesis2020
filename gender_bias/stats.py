from gender_bias.analysis import Analysis
from gender_bias.data_driver import DataDriver

language = 'en'
corpus_folder = '../translation/corpus/'
dd = DataDriver(corpus_folder=corpus_folder, languages={'en'}, save_dir='')
for corpus in ['biographies', 'balanced']:
    sentences = list(map(str.strip, dd.load_corpus(corpus)[language]))
    an = Analysis('en', corpus=corpus, filter_unbiased=True)
    an.print_gender_stats(sentences, corpus)
    # an.plot_gendered_vectors_by_pairs(n_neighbors=10, words_to_plot=20)
