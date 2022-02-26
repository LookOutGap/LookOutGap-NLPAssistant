"""
Copyright © 2020 Johnson & Johnson
"""

import warnings
from sklearn.metrics import pairwise_distances

supported_metrics = ['cosine', 'jaccard', 'manhattan', 'dice', 'hamming']

sparse_metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
dense_metrics = ['braycurtis', 'canberra', 'chebyshev', 'correlation',
                 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
                 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']


def similarity_calculation(new_mat,
                           old_mat,
                           metric: str = 'cosine'):
    """
    Calculate similarity be