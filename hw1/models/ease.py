"""EASE: exact L2-regularized linear reconstruction with a zero diagonal."""

import numpy as np
from scipy import sparse, linalg
from .base import Recommender


class EASE(Recommender):
    def __init__(self, regularization=100.0):
        self.regularization = regularization

    def fit(self, interactions):
        self.history = sparse.csr_matrix(interactions, dtype=np.float64)
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::len(gram) + 1] += self.regularization
        precision = linalg.cho_solve(linalg.cho_factor(gram, check_finite=False),
                                     np.eye(len(gram)), check_finite=False)
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0.0)
        return self

    def score(self, users):
        return self.history[users] @ self.weights
