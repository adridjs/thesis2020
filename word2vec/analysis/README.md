## Analysis
Gensim library implements most of the tools that are used to study gender bias, such as positive
 and negative contribution of word vectors to other vectors, which can be used to solve analogy queries such as:
  
 ```A is to B as C is to ?```
 
Given `A=King`, `B=Man`, `C=Queen` the result (with a properly trained word2vec model) would be `Woman`.

This will be used to evaluate the gender bias on our trained model 