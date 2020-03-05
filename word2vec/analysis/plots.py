import numpy as np
from bokeh.plotting import figure, show, output_file

from gensim.models import KeyedVectors


def load_embeddings(embeddings_filename, normalize=True, binary=False):
    kv = KeyedVectors.load_word2vec_format(embeddings_filename, binary=binary)
    print(kv.most_similar(positive=['girl'], negative=['guy']))

    embeddings = {word: emb/np.linalg.norm(emb) if normalize else emb for word, emb in zip(kv.index2word, kv.vectors)}

    return embeddings


def get_gender_vector(embeddings):
    guy = embeddings['guy']
    girl = embeddings['girl']

    gender_vector = guy - girl

    return gender_vector


def plot_gendered_vectors(gendered_vectors, threshold=None):
    output_file("line.html")
    if threshold:
        words, values = zip(*list(filter(lambda x: abs(x[1]) > threshold, gendered_vectors)))
    else:
        words, values = zip(*gendered_vectors)

    p = figure(x_range=words, y_range=(-1, 1))
    p.xaxis.major_label_orientation = np.pi / 4
    p.vbar(x=words, bottom=0, top=values, width=0.2)

    show(p)


if __name__ == '__main__':
    embs = load_embeddings('biographies_word2vec_5.txt')
    gen_vec = get_gender_vector(embs)

    # cos(u,v) =u·v / ‖u‖‖v‖
    # cos(w1, w2) = w1·w2 if embeddings are normalized between
    tokens = embs.keys()
    cos_similarity = [(word, np.dot(embs[word], gen_vec)) for word in tokens]

    gendered_words = sorted(cos_similarity, key=lambda x: x[1])
    for th in [0.1, 0.2, 0.3, 0.4, 0.45]:
        plot_gendered_vectors(gendered_words, threshold=th)
