import ast

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from utils.preprocessing import IndustryGrouper


if __name__ == "__main__":
    # NOTE loading dataset

    companies_df = pd.read_csv("../data/companies.csv")

    # initial preprocessing

    companies_df["business_tags"] = companies_df["business_tags"].apply(ast.literal_eval)
    companies_df["description"] = companies_df["description"].fillna("")
    companies_df = companies_df[companies_df["category"].notna()].reset_index(drop=True)

    # NOTE construct and classification pipelines

    construct_pipeline = Pipeline([
        ("grouper", IndustryGrouper()),
        ("transformer", ColumnTransformer([
                ("vectorizer", TfidfVectorizer(), "words")
        ])),
        ("clusterer", KMeans(220))
    ])

    classification_pipeline = Pipeline([
        ("grouper", IndustryGrouper()),
        ("transformer", ColumnTransformer([
                ("vectorizer", TfidfVectorizer(), "words")
        ])),
        ("clf", LogisticRegression())
    ])

    # NOTE training

    companies_df["taxonomy"] = construct_pipeline.fit_predict(companies_df)

    X_train, X_test, t_train, t_test = train_test_split(companies_df.drop(["taxonomy"], axis=1), companies_df["taxonomy"])

    classification_pipeline.fit(X_train, t_train)

    y_test = classification_pipeline.predict(X_test)

    # NOTE scores

    print(classification_report(t_test, y_test, zero_division=0))
