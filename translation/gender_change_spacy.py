import spacy
import re

from word2vec.data_driver import DataDriver


def get_gendered_words(docs):
    gendered_words = dict()
    for lang_gender, sentences_by_lang in docs.items():
        print(f'Processing language and gender: {lang_gender}')
        gendered_words[lang_gender] = set()
        for sentence in sentences_by_lang:
            doc = es_nlp(' '.join(sentence))
            for token in doc:
                tag = token.tag_
                pos = token.pos_
                m = re.match(r'.*Gender=(\w*)', tag)
                if m and pos in ['ADJ', 'DET']:
                    # gender = m.groups()
                    gendered_words[lang_gender].add(token.text)
                    # TODO:
    return gendered_words

es_nlp = spacy.load('es_core_news_sm')
#en_nlp = spacy.load('en_core_news_sm')
dd = DataDriver('../word2vec/biographies/', languages={'es'})
docs, _ = dd._parse_filtered_docs()
# TODO: Insert this DataDriver.get_balanced_dataset() in order to have a uniform distribution of sentences between genders.
gend_words = get_gendered_words(docs)
print(gend_words)
