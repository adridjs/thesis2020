GENDERS = {'he', 'she'}
LANGUAGES = {'en', 'es'}

# Choose any model you want for each language, these are from spacy and only used to clean the sentence from useless characters
# and normalizing it  (verb tenses, honorifics, stop words) in order to have a semantically sparse word vector space
NLP_MODELS = {'en': 'en_core_web_sm', 'es': 'es_core_news_sm'}

