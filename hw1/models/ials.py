"""Weighted low-rank least squares, solved by alternating normal equations."""

import numpy as np
from scipy import sparse, linalg
from .base import Recommender


class IALS(Recommender):
    def __init__(self, factors=32, regularization=1.0, alpha=20.0,
                 iterations=12, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def _solve_side(self, targets, observations, fixed):
        gram = fixed.T @ fixed
        gram.flat[::self.factors + 1] += self.regularization
        result = np.zeros((observations.shape[0], self.factors))
        
        for row in range(observations.shape[0]):
            start, stop = observations.indptr[row:row + 2]
            indices = observations.indices[start:stop]
            vectors = fixed[indices]
            extra = self.alpha * observations.data[start:stop]
            a = gram + (vectors.T * extra) @ vectors
            t_start, t_stop = targets.indptr[row:row + 2]
            t_indices = targets.indices[t_start:t_stop]
            if not len(t_indices):
                continue
            confidence = 1.0 + self.alpha * observations[row, t_indices].toarray().ravel()
            b = fixed[t_indices].T @ (confidence * targets.data[t_start:t_stop])
            result[row] = linalg.cho_solve(linalg.cho_factor(a, check_finite=False),
                                          b, check_finite=False)
        return result

    def fit(self, interactions, targets=None, observations=None):
        targets = sparse.csr_matrix(interactions if targets is None else targets,
                                    dtype=np.float64)
        observations = sparse.csr_matrix(interactions if observations is None
                                         else observations, dtype=np.float64)
        targets.sort_indices()
        observations.sort_indices()
        if targets.shape != observations.shape or np.any(observations.data < 0):
            raise ValueError("Invalid target/confidence matrices")
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, .05, (targets.shape[0], self.factors))
        self.item_factors = rng.normal(0, .05, (targets.shape[1], self.factors))
        for _ in range(self.iterations):
            self.user_factors = self._solve_side(targets, observations, self.item_factors)
            self.item_factors = self._solve_side(targets.T.tocsr(),
                                                 observations.T.tocsr(), self.user_factors)
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
