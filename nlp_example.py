"""
Copyright © 2020 Johnson & Johnson
"""

import pandas as pd
from nlprov.preprocessing import preprocess_text
from nlprov.vectorize import vectorize_text, vectorize_new_text
from nlprov.similarity_calc import similarity_calculation

text = pd.Series(data=["  Combination  of   spaces.    ",
                       "MixEd CASe",
                       ",./;'[]\-=",
  