"""
Copyright © 2020 Johnson & Johnson
"""

import pytest
import pandas as pd
import numpy as np
from nlprov.preprocessing import preprocess_text
from conftest import sents_chars_expected, sents_nums_expected, \
    sents_all_expected


# Creating data for the whitespace removal test
@pytest.fixture
def whitespace_removal_expected():
    return pd.Series(data=["front of line extra space.",
                           "mo