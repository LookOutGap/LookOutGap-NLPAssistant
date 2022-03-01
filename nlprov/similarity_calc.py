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
    Calculate similarity between two sparse document-feature matrices
    representing the new document-feature matrix (1 x f)
    and the old document-feature matrix(d x f)

    :param new_mat: scipy csr object of dimensions d x 