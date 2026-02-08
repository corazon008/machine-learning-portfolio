import pytest
from scipy.sparse import csr_matrix

from nlp.vectorizers import TfidfVectorizerWrapper


@pytest.fixture
def sample_texts():
    return [
        "this movie was great",
        "this movie was terrible",
        "absolutely fantastic movie",
    ]


def test_fit_transform_basic(sample_texts):
    vec = TfidfVectorizerWrapper(min_df=1, max_df=1.0)

    X = vec.fit_transform(sample_texts)

    # invariant: output type
    assert isinstance(X, csr_matrix)

    # invariant: correct number of rows
    assert X.shape[0] == len(sample_texts)

    # invariant: vocabulary not empty
    assert X.shape[1] > 0


def test_transform_before_fit_raises(sample_texts):
    vec = TfidfVectorizerWrapper(min_df=1)

    with pytest.raises(RuntimeError):
        vec.transform(sample_texts)


def test_double_fit_raises(sample_texts):
    vec = TfidfVectorizerWrapper(min_df=1)

    vec.fit(sample_texts)

    with pytest.raises(RuntimeError):
        vec.fit(sample_texts)


def test_consistent_dimensions_between_train_and_test():
    train = [
        "this movie was great",
        "this movie was terrible",
    ]
    test = [
        "this movie was great",
        "terrible movie",
    ]

    vec = TfidfVectorizerWrapper(min_df=1, max_df=1.0)

    X_train = vec.fit_transform(train)
    X_test = vec.transform(test)

    # invariant: same feature space
    assert X_train.shape[1] == X_test.shape[1]
