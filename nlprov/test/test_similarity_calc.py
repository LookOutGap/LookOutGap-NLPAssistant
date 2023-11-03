
"""
Copyright © 2020 Johnson & Johnson
"""

import pytest
from scipy.sparse import csr_matrix
from numpy import array, allclose
from nlprov.similarity_calc import similarity_calculation

# Set up data for testing similarity calculation
x = csr_matrix([0, 1, 1])
y = csr_matrix([1, 1, 1])

test_cases = [
        (x, y, 'cosine', 1 - array([0.18350342])),
        (x, y, 'jaccard', 1 - array([1/3])),
        (x, y, 'manhattan', 1 - array([1])),
        (x, y, 'dice', 1 - array([0.2])),
        (x, y, 'hamming', 1 - array([1/3
                                     ]))
]


# Test similarity calculation
@pytest.mark.parametrize(
        "x,y,metric,dist_expected",
        test_cases,