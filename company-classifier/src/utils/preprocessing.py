import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import TfidfVectorizer


class DebugTransformer(BaseEstimator, TransformerMixin):
    """
    Prints X for debbuging purposes
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        print(X)

        return X


class IndustryGrouper(BaseEstimator, TransformerMixin):
    """
    Concatenates `sector`, `category` and `niche` features into one string used
    for vectorization and drops the other features
    """
    
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        texts: list[str] = [" ".join([entry["sector"], entry["category"], entry["niche"]]) for entry in X.iloc]

        X = X.drop(["description", "business_tags", "sector", "category", "niche"], axis=1)

        X["words"] = texts

        return X


class DescriptionGrouper(BaseEstimator, TransformerMixin):
    """
    Concatenates `business_tags` and `description` features into one string used
    for vectorization and drops the other features
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        texts: list[str] = [" ".join(entry["business_tags"] + [entry["description"]]) for entry in X.iloc]

        X = X.drop(["description", "business_tags", "sector", "category", "niche"], axis=1)

        X["words"] = texts

        return X


class TextGrouper(BaseEstimator, TransformerMixin):
    """
    Concatenates all features into one string used for vectorization
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        texts: list[str] = [" ".join(entry["business_tags"] + [entry["description"], entry["sector"], entry["category"], entry["niche"]]) for entry in X.iloc]

        X = X.drop(["description", "business_tags", "sector", "category", "niche"], axis=1)

        X["words"] = texts

        return X


class IndustrySectorGrouper(BaseEstimator, TransformerMixin):
    """
    Keeps `sector` feature, concatenates `category` and `niche` features into one
    string used for vectorization and drops the other features
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        texts: list[str] = [" ".join([entry["category"], entry["niche"]]) for entry in X.iloc]

        X = X.drop(["description", "business_tags", "category", "niche"], axis=1)

        X["words"] = texts

        return X
    

class DescriptionSectorGrouper(BaseEstimator, TransformerMixin):
    """
    Keeps `sector` feature, concatenates `business_tags` and `description`
    features into one string used for vectorization and drops the other features
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        texts: list[str] = [" ".join(entry["business_tags"] + [entry["description"]]) for entry in X.iloc]

        X = X.drop(["description", "business_tags", "category", "niche"], axis=1)

        X["words"] = texts

        return X


class ExtendedSectorGrouper(BaseEstimator, TransformerMixin):
    """
    Keeps `sector` feature, concatenates all features into one string used for
    vectorization and drops the other features
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        texts: list[str] = [" ".join(entry["business_tags"] + [entry["description"], entry["category"], entry["niche"]]) for entry in X.iloc]

        X = X.drop(["description", "business_tags", "category", "niche"], axis=1)

        X["words"] = texts

        return X


class TaxonomyVectorizer(BaseEstimator, TransformerMixin):
    """
    Pretrained vectorizer using the taxonomy classes as documents
    """

    def __init__(self):
        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english")

        self._tfidf_vectorizer.fit(pd.read_csv("../data/insurance_taxonomy.csv")["label"])

    def fit(self, texts: list[str], y=None):
        return self

    def transform(self, texts: list[str]):
        return self._tfidf_vectorizer.transform(texts)
