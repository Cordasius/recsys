"""Item-based cosine CF with co-occurrence shrinkage and sparse neighbors."""

import numpy as np
from scipy import sparse
from .base import Recommender


class ItemKNN(Recommender):
    def __init__(self, neighbors=50, shrinkage=10.0):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, interactions):
        self.history = sparse.csr_matrix(interactions, dtype=np.float64)
        gram = (self.history.T @ self.history).toarray()
        norms = np.sqrt(np.diag(gram))
        denominator = norms[:, None] * norms[None, :]
        similarity = np.divide(gram, denominator, out=np.zeros_like(gram),
                               where=denominator > 0)
        if self.shrinkage:
            similarity *= gram / (gram + self.shrinkage)
        np.fill_diagonal(similarity, 0.0)
        
        sources, targets, values = [], [], []
        for j in range(similarity.shape[1]):
            order = np.argsort(-similarity[:, j], kind="stable")[:self.neighbors]
            order = order[similarity[order, j] > 0]
            sources.extend(order)
            targets.extend([j] * len(order))
            values.extend(similarity[order, j])
        self.weights = sparse.csr_matrix((values, (sources, targets)), shape=gram.shape)
        return self

    def score(self, users):
        return (self.history[users] @ self.weights).toarray()
