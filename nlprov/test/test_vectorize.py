"""
Copyright © 2020 Johnson & Johnson
"""

import pytest
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nlprov.vectorize import vec