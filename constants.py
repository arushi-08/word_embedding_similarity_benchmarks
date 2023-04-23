import os
import nltk

PATH_TO_FREQUENCIES_FILE = "data/sentence_similarity/frequencies.tsv"
PATH_TO_DOC_FREQUENCIES_FILE = "data/sentence_similarity/doc_frequencies.tsv"
STOP = set(nltk.corpus.stopwords.words("english"))
NUM_DOCS = 1288431

PATH_TO_WORD2VEC = os.path.expanduser("~/Downloads/GoogleNews-vectors-negative300.bin")
PATH_TO_GLOVE = os.path.expanduser("~/web_data/embeddings/glove.6B/glove.6B.300d.txt")
PATH_TO_FASTTEXT = os.path.expanduser("~/Downloads/wiki.en.vec")
TMP_GLOVE_FILE = "/tmp/glove.6B.300d.w2v.txt"