from gensim.models import Word2Vec

from extract_words import biography_sentences


model = Word2Vec(min_count=2, window=3)
model.build_vocab(biography_sentences)  # prepare the model vocabulary
model.train(biography_sentences, total_examples=model.corpus_count, epochs=model.iter)  # train word vectors
model.wv.save_word2vec_format("biographies_word2vec.txt", binary=False)
