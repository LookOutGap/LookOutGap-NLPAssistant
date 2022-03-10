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

    :param new_mat: scipy csr object of dimensions d x f for d documents and
        f features representing the new document-feature matrix, it should
        always be 1 x f.
    :param old_mat: scipy csr object of dimensions d x f for d documents and
        f features representing the old document-feature matrix, it should
        always be d x f.
    :param metric: string indicating the similarity/distance metric to be used,
        cosine is the default.

    :return: ndarray, similarity for each old document.
    """

    # Check that metric is supported
    assert metric in supported_metrics

    # Check dimensionality of new and old are compatible
    assert new_mat.shape[1] == old_mat.shape[1]

    # Check that new only contains a single nc
    assert new_mat.shap