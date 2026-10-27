"""Shared deterministic top-K ranking; model classes only produce scores."""

import numpy as np
from scipy import sparse


class Recommender:
    def score(self, users):
        raise NotImplementedError

    def recommend(self, users, seen, k=10):
        users = np.asarray(users, dtype=np.int64)
        scores = np.asarray(self.score(users), dtype=np.float64)
        if scores.shape != (len(users), seen.shape[1]):
            raise ValueError("Unexpected score shape")
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite model scores")
        mask = sparse.csr_matrix(seen)[users].tocoo()
        scores[mask.row, mask.col] = -np.inf
        if np.any(np.isfinite(scores).sum(axis=1) < k):
            raise ValueError("Fewer than k unseen candidates")
        
        return np.argsort(-scores, axis=1, kind="stable")[:, :k]
