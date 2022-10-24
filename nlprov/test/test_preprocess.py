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
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           ""
                           ])


# Testing the default regex preprocess
def test_regex_default(sents_regex, sents_default_expected):
    sents_preprocessed = preprocess_text(sents_regex,
                                         lowercase=False,
                                         eng_lang=False)
    pd.testing.assert_series_equal(sents_default_expected, sents_preprocessed)


# Unfortunately, you can't use fixtures in the parametrize so we are pulling
# the data from conftest.py
# Parametrize allows you to send through multiple parameters into the same test
# Testing a character regex, number regex, and all regex
@pytest.mark.parametrize("expected, regex",
                         [(sents_chars_expected(), '(?![A-Za-z]).'),
                          (sents_nums_expected(), '(?![0-9]).'),
                          (sents_all_expected(), '(?!).*')])
def test_regex_cases(sents_regex, expected, regex):
    sents_preprocessed = preprocess_text(sents_regex, lowercase=False,
                                         regex=regex,
                                         eng_lang=False)
    pd.testing.assert_series_equal(expected, sents_preprocessed)


# Creating data for the dict_replace test
@pytest.fixture
def sample_dict():
    return dict({'old': 'new',
                 'term': 'word'})


@pytest.fixture
def dict_replace_actual():
    return pd.Series(data=["no\xa0break space",
                           "old term",
                           "normal string"])


@pytest.fixture
def dict_replace_default_expected():
    return pd.Series(data=["no break space",
                           "old term",
                           "normal string"])


@pytest.fixture
def dict_replace_expected():
    return pd.Series(data=["no break space",
                           "new word",
                           "normal string"])


# Testing dict_replace with no additional dictionary items
def test_default_dict_replace(dict_replace_actual,
                              dict_replace_default_expected):
    preprocessed = preprocess_text(dict_replace_actual, eng_lang=False)
    pd.testing.assert_series_equal(dict_replace_default_expected,
                                   preprocessed)


# Testing dict_replace with additional dictionary items
def test_dict_replace(dict_replace_actual, dict_replace_expected, sample_dict):
    preprocessed = preprocess_text(dict_replace_actual,
                                   replace_dict=sample_dict,
                                   eng_lang=False)
    pd.testing.assert_series_equal(dict_replace_expected, preprocessed)


@pytest.fixture
def nan_removal_actual():
    return pd.Series(data=["front of line extra space",
                           np.nan,
                           pd.NaT])


@pytest.fixture
def nan_removal_expected():
    return pd.Series