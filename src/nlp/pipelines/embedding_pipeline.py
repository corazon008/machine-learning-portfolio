import numpy as np
from nlp.models.linear import LogisticRegressionModel
from nlp.data.preprocessing import TextPreprocessor
from nlp.features.embeddings import GensimWord2Vec, SentenceEmbedder


class EmbeddingPipeline:
    def __init__(self,
                 preprocessor: TextPreprocessor,
                 w2v: GensimWord2Vec,
                 embedder: SentenceEmbedder,
                 model: LogisticRegressionModel,
                 evaluator=None,
                 pretrained_w2v=False):
        self.preprocessor = preprocessor
        self.embedding_model = w2v
        self.embedder = embedder
        self.model = model
        self.evaluator = evaluator  # Here for future extensibility, currently not used since evaluate_model is a standalone function
        self.pretrained_w2v = pretrained_w2v

    def fit(self, X_train, y_train):
        """
        Raw text -> preprocess -> vectorize -> fit model
        :param X_train: Raw text data for training
        :param y_train: Labels for training data
        :return:
        """
        X_train_tokens = self.preprocessor.tokenize_batch(X_train)

        if not self.pretrained_w2v:
            if self.embedding_model.is_trained():
                raise RuntimeError("Embedding model is already trained. Cannot fit pipeline again.")
            self.embedding_model.train(X_train_tokens)

        self.embedder.__init__(self.embedding_model)

        X_train_embeddings = self.embedder.encode_corpus(X_train_tokens)
        self.model.fit(X_train_embeddings, y_train)

    def predict(self, X):
        if not self.pretrained_w2v:
            if not self.embedding_model.is_trained():
                raise RuntimeError("Embedding model must be trained before prediction. Call fit() first.")
        X_tokens = self.preprocessor.tokenize_batch(X)
        X_embeddings = self.embedder.encode_corpus(X_tokens)
        return self.model.predict(X_embeddings)

    def evaluate(self, X_test, y_test):
        X_test_tokens = self.preprocessor.tokenize_batch(X_test)
        X_test_embeddings = self.embedder.encode_corpus(X_test_tokens)
        return self.evaluator(self.model, X_test_embeddings, y_test)