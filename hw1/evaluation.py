"""Common full-catalog warm-start ranking protocol and validation search."""

import json
from time import perf_counter
import numpy as np
import pandas as pd
from models import ItemKNN, IALS, EASE, FactorizationMachine, CustomTargetALS
from models.base import Recommender
from data_utils import interaction_matrix


class Popularity(Recommender):
    def fit(self, interactions):
        self.counts = np.asarray(interactions.sum(axis=0)).ravel()
        return self

    def score(self, users):
        return np.broadcast_to(self.counts, (len(users), len(self.counts))).copy()


def make_relevance(heldout, seen, shape):
    relevant = interaction_matrix(heldout, shape)
    relevant = relevant - relevant.multiply(seen)
    relevant.eliminate_zeros()
    
    users = np.flatnonzero(np.diff(relevant.indptr) > 0)
    users = users[np.diff(seen.indptr)[users] <= shape[1] - 10]
    if not len(users):
        raise ValueError("No eligible held-out users")
    return relevant.tocsr(), users


def evaluate(model, relevant, users, seen, k=10, batch_size=256):
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(discounts)
    rows, recommended = [], set()
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        top = model.recommend(batch, seen, k)
        for u, ranking in zip(batch, top):
            truth = relevant.getrow(u).indices
            hits = np.isin(ranking, truth).astype(float)
            dcg = hits @ discounts
            rows.append((u, hits.sum() / len(truth), dcg / ideal[min(k, len(truth)) - 1],
                         hits.sum() / k, float(hits.any()), len(truth)))
            recommended.update(ranking.tolist())
    details = pd.DataFrame(rows, columns=["user_idx", "Recall@10", "NDCG@10",
                                         "Precision@10", "HR@10", "relevant_items"])
    means = details[["Recall@10", "NDCG@10", "Precision@10", "HR@10"]].mean().to_dict()
    means.update(coverage=len(recommended) / seen.shape[1], eval_users=len(users),
                 candidate_items=seen.shape[1], relevant_pairs=int(details.relevant_items.sum()))
    return means, details


def fit_model(name, parameters, history, shape, metadata_codes, cutoff):
    positive = interaction_matrix(history, shape)
    seen = interaction_matrix(history, shape, positive_only=False)
    start = perf_counter()
    if name == "Popularity":
        model = Popularity().fit(positive)
    elif name == "ItemKNN":
        model = ItemKNN(**parameters).fit(positive)
    elif name == "iALS":
        model = IALS(**parameters).fit(positive)
    elif name == "EASE":
        model = EASE(**parameters).fit(positive)
    elif name.startswith("FM"):
        model = FactorizationMachine(**parameters, use_metadata=(name == "FM + metadata"))
        model.fit(positive, metadata_codes, seen)
    elif name == "Custom ALS":
        model = CustomTargetALS(**parameters).fit(positive, history, cutoff)
    else:
        raise ValueError(name)
    model.fit_seconds = perf_counter() - start
    return model


def search(name, grid, history, shape, metadata_codes, cutoff, relevant, users, seen):
    rows = []
    for parameters in grid:
        model = fit_model(name, parameters, history, shape, metadata_codes, cutoff)
        metrics, _ = evaluate(model, relevant, users, seen)
        rows.append({"model": name, "parameters": json.dumps(parameters, sort_keys=True),
                     "fit_seconds": model.fit_seconds, **metrics})
        print(f"{name}: {parameters}; validation NDCG@10={metrics['NDCG@10']:.5f}", flush=True)
    table = pd.DataFrame(rows)
    
    best = int(np.argmax(table["NDCG@10"].to_numpy()))
    return dict(grid[best]), table


def paired_interval(left, right, seed=42, repetitions=2000):
    paired = left[["user_idx", "NDCG@10"]].merge(
        right[["user_idx", "NDCG@10"]], on="user_idx", suffixes=("_a", "_b"),
        validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError("Different evaluation users")
    difference = paired["NDCG@10_a"].to_numpy() - paired["NDCG@10_b"].to_numpy()
    rng = np.random.default_rng(seed)
    bootstrap = np.array([rng.choice(difference, len(difference), replace=True).mean()
                          for _ in range(repetitions)])
    return {"delta_NDCG@10": float(difference.mean()),
            "ci_low": float(np.quantile(bootstrap, .025)),
            "ci_high": float(np.quantile(bootstrap, .975))}
