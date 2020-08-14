from gender_bias.analysis import Analysis

if __name__ == '__main__':
    analysis = Analysis('en')
    # Pca comparison between definitional pairs and random vectors
    # analysis.plot_pca()

    # Plot gendered word vectors
    analysis.plot_gendered_vectors_by_pairs()
