## Corpus Generator
Utility to generate the different corpus that are going to be analyzed. We will measure its gender bias both in the
 embeddings space after training a word2vec model and after translating sentences on a fairseq model.

Creates a mixed dataset between 2 corpus (can be easily adapted to more) in order to apply the Mixed Fine Tuning
 strategy that has become a good way to perform domain adaption on pretrained models. This method cope with the
  "catastrophic forgetting" problem, giving to the network out-of-domain (general domain which the transformer
   has been trained on) samples together with in-domain samples
   with in-domain samples. 
   
   The CorpusGenerator expects to have files in `data/` folder with the following pattern: 
   * For the general domain dataset, `corpus.clean.{language}` 
   * For the domain-specific dataset,  `{lang}_{gender}.txt` 
   
   The domain-specific dataset is the one generated by `word2vec.DataDriver.save_sentences`
