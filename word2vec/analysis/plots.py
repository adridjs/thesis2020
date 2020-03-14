import numpy as np
from bokeh.plotting import figure, show, output_file

from gensim.models import KeyedVectors


def load_embeddings(embeddings_filename, normalize=True, binary=False):
    kv = KeyedVectors.load_word2vec_format(embeddings_filename, binary=binary)
    print(kv.most_similar(positive=['girl'], negative=['guy']))

    embeddings = {word: emb/np.linalg.norm(emb) if normalize else emb for word, emb in zip(kv.index2word, kv.vectors)}

    return embeddings


def plot_gendered_vectors(gendered_vectors, pair=None, threshold=None):
    output_file("line.html")
    if threshold:
        words, values = zip(*list(filter(lambda x: abs(x[1]) > threshold, gendered_vectors)))
    else:
        words, values = zip(*gendered_vectors)

    p = figure(title=f'{pair}', x_range=words, y_range=(-1, 1))
    p.xaxis.major_label_orientation = np.pi / 4
    p.vbar(x=words, bottom=0, top=values, width=0.2)

    show(p)


if __name__ == '__main__':
    embs = load_embeddings('word2vec/biographies_word2vec_5.txt')
    print(f'Vocabulary size: {len(embs)}')

    # definitional pairs
    pairs = [('he', 'she'), ('father', 'mother'), ('his', 'her'), ('man', 'woman'), ('boy', 'girl')]

    # Based on arxiv:1903.03862v2
    for masc, fem in pairs:
        gender_vector = embs[masc] - embs[fem]
        # cos(u,v) =u·v / ‖u‖‖v‖
        # cos(w1, w2) = w1·w2 if embeddings are normalized between
        professions = list(map(str.strip, open('word2vec/professions.txt').readlines()))
        profession_embeddings = {token: emb for token, emb in embs.items() if token in professions}
        print(f'Number of professions in vocabulary: {len(profession_embeddings)}')
        cos_similarity = [(word, np.dot(embs[word], gender_vector)) for word in profession_embeddings]

        gendered_words = sorted(cos_similarity, key=lambda x: x[1])
        for th in [0.05, 0.1, 0.15, 0.2]:
            try:
                plot_gendered_vectors(gendered_words, pair=(masc, fem), threshold=th)
            except ValueError:
                pass
