"""
Copyright © 2020 Johnson & Johnson
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def vectorize_text(text_col: pd.Series,
                   vec_type: str = 'count',
                   **kwargs):
    """
    Vectoriz