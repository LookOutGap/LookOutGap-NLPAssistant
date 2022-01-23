"""
Copyright © 2020 Johnson & Johnson
"""

import en_core_web_sm


def get_spacy_nlp():
    try:
        spacy_nlp = en_core_web_sm.load(disable=['parser'