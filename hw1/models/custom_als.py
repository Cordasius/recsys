"""ALS with signed, recency-decayed preferences instead of binary relevance."""

import numpy as np
from scipy import sparse
from .ials import IALS


class CustomTargetALS(IALS):
    def __init__(self, half_life_days=365.0, **kwargs):
        super().__init__(**kwargs)
        self.half_life_days = half_life_days

    def make_targets(self, history, shape, cutoff):
        age_days = (float(cutoff) - history.timestamp.to_numpy()) / 86_400_000
        if np.any(age_days < 0):
            raise ValueError("Future events in training history")
        value = ((history.rating.to_numpy() - 3.0) / 2.0
                 * np.exp2(-age_days / self.half_life_days))
        return sparse.csr_matrix((value, (history.user_idx, history.item_idx)), shape=shape)

    def fit(self, interactions, history, cutoff):
        # Same confidence as baseline: 1 + alpha * 1[rating >= 4].
        # Low ratings change p, not confidence. Unobserved entries have p=0,c=1.
        targets = self.make_targets(history, interactions.shape, cutoff)
        return super().fit(interactions, targets=targets, observations=interactions)
