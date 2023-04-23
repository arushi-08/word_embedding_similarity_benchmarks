import os
import gensim

from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec

from constants import PATH_TO_WORD2VEC, PATH_TO_GLOVE, PATH_TO_FASTTEXT, TMP_GLOVE_FILE


class StaticEmbeddings:
    def init(self):
        self.word2vec = None
        self.glove = None
        self.fasttext = None

    def load_word2vec(self):
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_WORD2VEC, binary=True)
        return self.word2vec
    
    def load_glove(self):
        glove2word2vec(PATH_TO_GLOVE, TMP_GLOVE_FILE)
        self.glove = gensim.models.KeyedVectors.load_word2vec_format(TMP_GLOVE_FILE)
        return self.glove

    def load_fasttext(self):
        self.fasttext = gensim.models.KeyedVectors.load_word2vec_format(
            PATH_TO_FASTTEXT, binary=False
        )
        return self.fasttext