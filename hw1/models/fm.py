"""Second-order FM, BPR training with autograd and training-only negatives."""

import numpy as np
import torch
from scipy import sparse
from .base import Recommender


class FactorizationMachine(Recommender):
    def __init__(self, factors=32, learning_rate=.005, regularization=1e-5,
                 epochs=12, batch_size=2048, use_metadata=True, seed=42):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_metadata = use_metadata
        self.seed = seed

    def _features(self, users, items):
        u = torch.as_tensor(users, dtype=torch.long).reshape(-1, 1)
        i = torch.as_tensor(items, dtype=torch.long)
        return torch.cat([u, self.item_features[i]], dim=1)

    def _forward(self, features):
        vectors = self.embedding(features)
        return (self.linear(features).sum(dim=(1, 2))
                + .5 * (vectors.sum(dim=1).square()
                        - vectors.square().sum(dim=1)).sum(dim=1))

    def fit(self, interactions, metadata_codes, seen):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        interactions = sparse.csr_matrix(interactions)
        seen = sparse.csr_matrix(seen)
        n_users, self.n_items = interactions.shape
        blocks = [np.arange(self.n_items, dtype=np.int64) + n_users]
        offset = n_users + self.n_items
        if self.use_metadata:
            for column in np.asarray(metadata_codes).T:
                if column.min() < 0:
                    raise ValueError("Metadata codes must be nonnegative")
                blocks.append(column + offset)
                offset += int(column.max()) + 1
        self.item_features = torch.as_tensor(np.stack(blocks, axis=1), dtype=torch.long)
        self.embedding = torch.nn.Embedding(offset, self.factors)
        self.linear = torch.nn.Embedding(offset, 1)

        initial = rng.normal(0, .03, (offset, self.factors)).astype(np.float32)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.from_numpy(initial))
        torch.nn.init.zeros_(self.linear.weight)
       
        rng = np.random.default_rng(self.seed)
        optimizer = torch.optim.Adam([*self.embedding.parameters(), *self.linear.parameters()],
                                     lr=self.learning_rate, weight_decay=self.regularization)
        users, items = interactions.nonzero()
        usable = np.diff(seen.indptr)[users] < self.n_items
        users, items = users[usable], items[usable]
        if not len(users):
            raise ValueError("No positive pairs with available negative candidates")
        self.loss_history = []
        for _ in range(self.epochs):
            order = rng.permutation(len(users))
            loss_sum = 0.0
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                u, pos = users[idx], items[idx]
                neg = rng.integers(self.n_items, size=len(idx))
                collision = np.asarray(seen[u, neg]).ravel() != 0
                while collision.any():
                    neg[collision] = rng.integers(self.n_items, size=collision.sum())
                    collision = np.asarray(seen[u, neg]).ravel() != 0
                delta = self._forward(self._features(u, pos)) - self._forward(self._features(u, neg))
                loss = torch.nn.functional.softplus(-delta).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(idx)
            self.loss_history.append(loss_sum / len(order))
        self._cache_scores(n_users)
        return self

    @torch.no_grad()
    def _cache_scores(self, n_users):
        self.user_vectors = self.embedding.weight[:n_users].numpy().copy()
        self.user_bias = self.linear.weight[:n_users, 0].numpy().copy()
        vectors = self.embedding(self.item_features)
        self.item_vectors = vectors.sum(dim=1).numpy()
        self.item_bias = (self.linear(self.item_features).sum(dim=(1, 2))
                          + .5 * (vectors.sum(dim=1).square()
                                  - vectors.square().sum(dim=1)).sum(dim=1)).numpy()

    def score(self, users):
        return (self.user_vectors[users] @ self.item_vectors.T
                + self.user_bias[users, None] + self.item_bias[None, :])
