# Study of Low Dimensional Embeddings

The code is organized in following files: \
word_similarity_benchmarks.py - runs the word similarity experiments using static and contextual embeddings. \
sentence_similarity_benchmarks.py - runs the sentence similarity experiments using static and contextual embeddings. \
bert_layer_wise_benchmark.py - runs the word similarity experiments using different BERT model's hidden layer outputs. \
data.py - helper script to load/download datasets. \
static_embedding.py - loads static word embeddings models.

## Requirements

* Python 3.8+

I have used the Gensim library to load static embeddings, HuggingFace library to get the BERT, GPT-2, Allennlp for ELMo and lastly, sentence-transformers library for Sentence-BERT and GPT-2 models.


### References used : 
https://github.com/kudkudak/word-embeddings-benchmarks
https://github.com/nlptown/nlp-notebooks/blob/master/Simple%20Sentence%20Similarity.ipynb
