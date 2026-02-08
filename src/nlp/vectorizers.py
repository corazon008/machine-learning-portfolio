from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


class TfidfVectorizerWrapper:
    """
    Thin wrapper around sklearn TfidfVectorizer to enforce
    correct usage in an ML pipeline.
    """

    def __init__(
            self,
            ngram_range: tuple[int, int] = (1, 2),
            max_features: Optional[int] = None,
            min_df: int = 2,
            max_df: float = 0.95,
    ):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df, # ignore terms that appear in fewer than min_df documents
            max_df=max_df, # ignore terms that appear in more than max_df fraction of documents
            norm="l2",
        )
        self._is_fitted = False

    def fit(self, texts: List[str]) -> None:
        if self._is_fitted:
            raise RuntimeError("Vectorizer is already fitted")

        self.vectorizer.fit(texts)
        self._is_fitted = True

    def transform(self, texts: List[str]) -> csr_matrix:
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fitted before calling transform")

        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        if self._is_fitted:
            raise RuntimeError("Vectorizer is already fitted")

        X = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        return X

    def get_feature_names(self) -> List[str]:
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fitted first")

        return self.vectorizer.get_feature_names_out().tolist()
