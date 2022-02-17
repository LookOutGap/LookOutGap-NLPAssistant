"""
Copyright © 2020 Johnson & Johnson
"""

import warnings
from sklearn.metrics import pairwise_distances

supported_metrics = ['cosine', 'jaccard', 'manhattan', 'dice', 'hamming']

sparse_metrics = ['cityblo