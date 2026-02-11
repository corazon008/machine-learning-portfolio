from typing import List, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    """
    Wrapper around sklearn LogisticRegression for NLP classification.
    """

    def __init__(
            self,
            C: float = 1.0,
            max_iter: int = 1000,
            random_state: int = 42,
    ):
        self.model = LogisticRegression(
            C=C,
            #penalty="l2",
            l1_ratio=0,
            solver="liblinear",
            max_iter=max_iter,
            random_state=random_state,
        )
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

    def score(self, X: csr_matrix, y: np.ndarray) -> float:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before scoring")

        return self.model.score(X, y)

    def get_top_features(
            self,
            feature_names: List[str],
            k: int = 20,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Return top-k positive and negative features based on learned weights.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        if len(feature_names) != self.model.coef_.shape[1]:
            raise ValueError("Feature names size does not match model coefficients")

        coefs = self.model.coef_[0]

        top_positive_idx = np.argsort(coefs)[-k:][::-1]
        top_negative_idx = np.argsort(coefs)[:k]

        top_positive = [(feature_names[i], coefs[i]) for i in top_positive_idx]
        top_negative = [(feature_names[i], coefs[i]) for i in top_negative_idx]

        return top_positive, top_negative
