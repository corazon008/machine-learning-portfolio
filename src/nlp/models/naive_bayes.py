import numpy as np
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB


class MultinomialNBModel:
    """
    Wrapper around sklearn Multinomial Naive Bayes for NLP classification.
    """

    def __init__(self, alpha: float = 1.0):
        self.model = MultinomialNB(alpha=alpha)
        self._is_fitted = False

    def fit(self, X: csr_matrix, y: np.ndarray) -> None:
        if self._is_fitted:
            raise RuntimeError("Model is already fitted")

        self.model.fit(X, y)
        self._is_fitted = True

    def predict(self, X: csr_matrix) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        return self.model.predict(X)

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        return self.model.predict_proba(X)