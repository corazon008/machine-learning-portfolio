from gensim.models import Word2Vec
import numpy as np


class GensimWord2Vec:
    def __init__(
            self,
            vector_size=100,
            window=5,
            min_count=5,
            sg=1,
            negative=5,
            workers=4,
    ):
        self.model = Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            negative=negative,
            workers=workers,
        )

    def train(self, tokenized_texts, epochs=5):
        self.model.build_vocab(tokenized_texts)
        self.model.train(
            tokenized_texts,
            total_examples=len(tokenized_texts),
            epochs=epochs,
        )

    def get_word_vector(self, word):
        return self.model.wv[word]

    def has_word(self, word):
        return word in self.model.wv

    def is_trained(self):
        return len(self.model.wv) > 0

class SentenceEmbedder:
    def __init__(self, w2v_model):
        self.w2v = w2v_model

    def encode_sentence(self, tokens):
        vectors = [
            self.w2v.get_word_vector(w)
            for w in tokens
            if self.w2v.has_word(w)
        ]

        if len(vectors) == 0:
            return np.zeros(self.w2v.model.vector_size)

        return np.mean(vectors, axis=0)

    def encode_corpus(self, tokenized_texts):
        return np.array(
            [self.encode_sentence(tokens) for tokens in tokenized_texts]
        )

class PretrainedSentenceEmbedder:
    def __init__(self, pretrained_model):
        self.model = pretrained_model
        self.vector_size = pretrained_model.vector_size

    def encode_sentence(self, tokens):
        vectors = [
            self.model[w]
            for w in tokens
            if w in self.model
        ]

        if len(vectors) == 0:
            return np.zeros(self.vector_size)

        return np.mean(vectors, axis=0)

    def encode_corpus(self, tokenized_texts):
        return np.array(
            [self.encode_sentence(tokens) for tokens in tokenized_texts]
        )
