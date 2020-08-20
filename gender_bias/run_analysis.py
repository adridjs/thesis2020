from gender_bias.analysis import Analysis

if __name__ == '__main__':
    analysis_balanced = Analysis('en', corpus='balanced')
    analysis_balanced.plot_pca()
    analysis_EP = Analysis('en', corpus='EuroParl')
    analysis_EP.plot_pca()
