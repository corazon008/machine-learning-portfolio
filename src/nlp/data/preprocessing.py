import re
from typing import List


class TextPreprocessor:
    """
    Text normalization for NLP baseline models.
    This class is deliberately simple and deterministic.
    """

    def __init__(self):
        # compile regex once for performance
        self._non_alphanum = re.compile(r"[^a-z0-9\s]")
        self._multiple_spaces = re.compile(r"\s+")
        self._html_tags = re.compile(r"<.*?>")

    def preprocess(self, text: str) -> str:
        """
        Apply basic normalization to a single text document.
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")

        # lowercase
        text = text.lower()

        # remove HTML tags
        text = self._html_tags.sub(" ", text)

        # remove punctuation / special characters
        text = self._non_alphanum.sub(" ", text)

        # normalize whitespace
        text = self._multiple_spaces.sub(" ", text)

        return text.strip()

    def transform(self, texts: List[str]) -> List[str]:
        """
        Alias for preprocess_batch to fit sklearn transformer interface.
        """
        return self.preprocess_batch(texts)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Apply preprocessing to a list of documents.
        """
        return [self.preprocess(t) for t in texts]

    def tokenize(self, text: str) -> List[str]:
        return self.preprocess(text).split()

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        return [self.tokenize(t) for t in texts]

