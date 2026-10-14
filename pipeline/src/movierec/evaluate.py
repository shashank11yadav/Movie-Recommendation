"""Offline evaluation: HitRate@k, NDCG@k, catalog coverage and popularity bias."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from movierec import config
from movierec.catalog import Catalog
from movierec.collab import interaction_matrix
from movierec.neighbors import FloatArray, IntArray
from movierec.scoring import Model, Scorer

# Share of the catalog, by training-rating count, that counts as "head" when
# measuring how much a model recommends outside the most popular titles.
HEAD_FRACTION = 0.2


HEAD_FRACTION = 0.2
BOOTSTRAP_SAMPLES = 2000


@dataclass(frozen=True)
class Metrics:
    hit_rate: float
    ndcg: float
    # 95% percentile-bootstrap intervals over users. With ~600 test users the
    # intervals are wide, and the README says so.
    hit_rate_ci: tuple[float, float]
    ndcg_ci: tuple[float, float]
    coverage: float
    avg_popularity: float
    long_tail_share: float
    users: int

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in asdict(self).items():
            if isinstance(value, float):
                out[key] = round(value, 4)
            elif isinstance(value, tuple):
                out[key] = [round(v, 4) for v in value]
            else:
                out[key] = value
        return out


def head_items(counts: IntArray, fraction: float = HEAD_FRACTION) -> IntArray:
    """Rows of the most-rated ``fraction`` of items that have at least one rating."""
    rated = np.flatnonzero(counts > 0)
    n_head = max(1, round(len(rated) * fraction))
    order = np.lexsort((rated, -counts[rated]))
    return rated[order[:n_head]].astype(np.int64)


def bootstrap_ci(per_user: FloatArray, samples: int = BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    """Percentile bootstrap of the mean. Seeded so results.json is reproducible."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(per_user), size=(samples, len(per_user)))
    means = per_user[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def compute_metrics(
    recs: list[IntArray],
    targets: IntArray,
    counts: IntArray,
    n_items: int,
    k: int = config.EVAL_CUTOFF,
) -> Metrics:
    head = np.zeros(n_items, dtype=bool)
    head[head_items(counts)] = True
    users = len(recs)
    hit = np.zeros(users, dtype=np.float64)
    gain = np.zeros(users, dtype=np.float64)
    pops, tails = [], []
    seen: set[int] = set()
    for u, (rec, target) in enumerate(zip(recs, targets.tolist(), strict=True)):
        top = rec[:k]
        seen.update(top.tolist())
        position = np.flatnonzero(top == target)
        if position.size:
            hit[u] = 1.0
            # One relevant item per user, so the ideal DCG is 1.
            gain[u] = 1.0 / np.log2(position[0] + 2)
        if top.size:
            pops.append(float(counts[top].mean()))
            tails.append(float((~head[top]).mean()))
    return Metrics(
        hit_rate=float(hit.mean()),
        ndcg=float(gain.mean()),
        hit_rate_ci=bootstrap_ci(hit),
        ndcg_ci=bootstrap_ci(gain),
        coverage=len(seen) / n_items,
        avg_popularity=float(np.mean(pops)) if pops else 0.0,
        long_tail_share=float(np.mean(tails)) if tails else 0.0,
        users=users,
    )


@dataclass(frozen=True)
class EvalData:
    """Everything needed to score one split, with matrix rows aligned to ``users``."""

    users: IntArray
    train: pd.DataFrame
    seen: sp.csr_array
    targets: IntArray

    @classmethod
    def build(cls, train: pd.DataFrame, test: pd.DataFrame, catalog: Catalog) -> EvalData:
        users = np.sort(test["userId"].unique()).astype(np.int64)
        train = train[train["userId"].isin(users)]
        by_user = test.set_index("userId").loc[users, "movieId"].to_numpy()
        return cls(
            users=users,
            train=train,
            seen=interaction_matrix(train, catalog, users),
            targets=catalog.rows_of(by_user),
        )

    def profile(self, catalog: Catalog, size: int | None) -> sp.csr_array:
        """Query matrix: each user's ``size`` most recent training ratings, or all of them."""
        if size is None:
            return self.seen
        recent = (
            self.train.sort_values(["userId", "timestamp", "movieId"], kind="stable")
            .groupby("userId")
            .tail(size)
        )
        return interaction_matrix(recent, catalog, self.users)


def evaluate(
    model: Model,
    scorer: Scorer,
    query: sp.csr_array,
    data: EvalData,
    k: int = config.EVAL_CUTOFF,
) -> Metrics:
    n_items = query.shape[1]
    if model == "popularity":
        base = scorer.counts.astype(np.float64)
        scores = np.broadcast_to(base, (query.shape[0], n_items))
    else:
        scores = (query @ scorer.item_matrix(model)).toarray()

    recs = []
    for u in range(query.shape[0]):
        start, end = data.seen.indptr[u], data.seen.indptr[u + 1]
        exclude = data.seen.indices[start:end].astype(np.int64)
        recs.append(scorer.rank(np.array(scores[u], dtype=np.float64), exclude, k))
    return compute_metrics(recs, data.targets, scorer.counts, n_items, k)
