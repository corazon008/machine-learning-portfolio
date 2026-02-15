from nlp.features.tfidf import TfidfVectorizerWrapper
from nlp.models.linear import LogisticRegressionModel
from nlp.models.naive_bayes import MultinomialNBModel
from nlp.data.preprocessing import TextPreprocessor
from nlp.evaluation.metrics import evaluate_model


class TFIDFPipeline:
    def __init__(self,
                 preprocessor: TextPreprocessor,
                 vectorizer: TfidfVectorizerWrapper,
                 model: LogisticRegressionModel | MultinomialNBModel,
                 evaluator=None):
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.model = model
        self.evaluator = evaluator  # Here for future extensibility, currently not used since evaluate_model is a standalone function

    def fit(self, X_train, y_train):
        """
        Raw text -> preprocess -> vectorize -> fit model
        :param X_train: Raw text data for training
        :param y_train: Labels for training data
        :return:
        """
        if self.vectorizer.is_fitted():
            raise RuntimeError("Vectorizer is already fitted. Cannot fit pipeline again.")

        X_clean = self.preprocessor.transform(X_train)
        X_vec = self.vectorizer.fit_transform(X_clean)
        self.model.fit(X_vec, y_train)

    def predict(self, X):
        if not self.vectorizer.is_fitted():
            raise RuntimeError("Vectorizer must be fitted before prediction. Call fit() first.")
        X_clean = self.preprocessor.transform(X)
        X_vec = self.vectorizer.transform(X_clean)
        return self.model.predict(X_vec)

    def evaluate(self, X_test, y_test):
        X_test = self.preprocessor.transform(X_test)
        X_test = self.vectorizer.transform(X_test)
        return evaluate_model(self.model, X_test, y_test)
