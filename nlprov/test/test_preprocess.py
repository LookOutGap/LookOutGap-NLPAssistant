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
                           "more space at front of line.",
                           "End of line extra space.",
                           "More space at end of line.",
                           "Space in the middle.",
                           "More space in the middle.",
                           "Combination of spaces."])


@pytest.fixture
def whitespace_removal_actual():
    return pd.Series(data=[" front of line extra space.",
                           "    more space at front of line.",
                           "End of line extra space. ",
                           "More space at end of line.     ",
                           "Space in the  middle.",
                           "More space in the   middle.",
                           "  Combination  of   spaces.    "])


# Testing whitespace removal
def test_whitespace_removal(whitespace_removal_actual,
                            whitespace_removal_expected):
    preprocessed = preprocess_text(whitespace_removal_actual, lowercase=False,
                                   regex='(?!).*', eng_lang=False)
    pd.testing.assert_series_equal(whitespace_removal_expected, preprocessed)


# Creating data for the lowercase test
@pytest.fixture
def lowercase_actual():
    return pd.Series(data=["ALL UPPERCASE",
                           "CaMeL cAsE",
                           "MixEd CASe",
                           "all lowercase"])


@pytest.fixture
def lowercase_expected():
    return pd.Series(data=["all uppercase",
                           "camel case",
                           "mixed case",
                           "all lowercase"])


# Testing lowercase
def test_lowercase(lowercase_actual, lowercase_expected):
    preprocessed = preprocess_text(lowercase_actual,
                                   regex='(?!).*',
                                   eng_lang=False)
    pd.testing.assert_series_equal(lowercase_expected, preprocessed)


# Some examples from:
# https://github.com/explosion/spaCy/blob/master/spacy/tests/tokenizer/test_naughty_strings.py
# Creating regex series that will be used for the next 4 tests
@pytest.fixture
def sents_regex():
    return pd.Series(data=["ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                           "abcdefghijklmnopqrstuvwxyz",
                           "0123456789",
                           "",
                           ",./;'[]\-=",
                           '<>?:"{}|_+',
                           '!@#$%^&*()`~"',
                           "Ω≈ç√∫˜µ≤≥÷",
                           "­؀؁؂؃؄؅؜۝܏᠎​‌‍‎‏‪",
                           "åß∂ƒ©˙∆˚¬…æ",
                           "œ∑´®†¥¨ˆøπ“‘",
                           "¡™£¢∞§¶•ªº–≠",
                           "¸˛Ç◊ı˜Â¯˘¿",
                           "ÅÍÎÏ˝ÓÔÒÚÆ☃",
                           "Œ„´‰ˇÁ¨ˆØ∏”’",
                           "`⁄€‹›ﬁﬂ‡°·‚—±",
                           "⅛⅜⅝⅞"
                           ])


# Expected response when default preprocessing
@pytest.fixture
def sents_default_expected():
    return pd.Series(data=["ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                           "abcdefghijklmnopqrstuvwxyz",
                           "0123456789",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                