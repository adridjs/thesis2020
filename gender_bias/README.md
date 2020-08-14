# Word2Vec
This module is used to train a word2vec model that generates word vectors in an embedded space,
 based on context-similar words.

## Word2Vec Trainer
```
python3 run_word2vec.py 
```

### Analysis

#### Analogies
Gensim library implements most of the tools that are used to study gender bias, such as positive
 and negative contribution of word vectors to other vectors, which can be used to solve analogy queries such as:
  
 ```A is to B as C is to ?```
 
Given `A=King`, `B=Man`, `C=Queen` the result (with a properly trained word2vec model) would be `Woman`.

This will be used to evaluate the gender bias on our trained model
#### Pearson Correlation

### TensorflowProjection

In a python3.5>= console, run
```
embeddings_vectors = np.stack(list(embeddings.values(), axis=0))
# shape [n_words, embedding_size]
emb = tf.Variable(embeddings_vectors, name='word_embeddings')

# Add an op to initialize the variable.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables and save the
# variables to disk.
with tf.Session() as sess:
   sess.run(init_op)

# Save the variables to disk.
   save_path = saver.save(sess, "model_dir/model.ckpt")
   print("Model saved in path: %s" % save_path)

words = '\n'.join(list(embeddings.keys()))

with open(os.path.join('model_dir', 'metadata.tsv'), 'w') as f:
   f.write(words)
```

Then in a bash console, run
```
tensorboard --logdir model_dir
```