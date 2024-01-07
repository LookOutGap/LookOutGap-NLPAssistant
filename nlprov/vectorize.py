"""
Copyright © 2020 Johnson & Johnson
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def vectorize_text(text_col: pd.Series,
                   vec_type: str = 'count',
                   **kwargs):
    """
    Vectorizes pre-processed text. Instantiates the vectorizer and
    fit_transform it to the data provided.

    :param text_col: Pandas series, containing preprocessed text.
    :param vec_type: string indicating what type of vectorization
        (count or tfidf currently).
    :param **kwargs: dict of keyworded arguments for sklearn vectorizer
        functions.

    :return: A tuple containing vectorized (doc-feature matrix that as d rows
        and f columns for count and tfidf vectorization) and vectorizer_obj
        (vectorization sklearn object representing trained vectorizer).
    """

    # Check if vectorization type is suppor