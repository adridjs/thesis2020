from gender_bias.analysis import Analysis
from gender_bias.data_driver import DataDriver

if __name__ == '__main__':
    language = 'en'
    corpus_folder = '../translation/corpus/'
    dd = DataDriver(corpus_folder=corpus_folder, languages={'en'})
    for corpus in ['balanced', 'EuroParl']:
        analysis = Analysis(language, corpus=corpus)
        analysis.plot_pca(sample=10)
