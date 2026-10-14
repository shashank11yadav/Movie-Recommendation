from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from movierec import collab
from movierec.catalog import Catalog
from movierec.evaluate import EvalData, bootstrap_ci, compute_metrics, evaluate, head_items
from movierec.neighbors import from_row_scores
from movierec.scoring import HybridParams, Scorer
from movierec.split import leave_last_out


def test_hit_rate_and_ndcg_by_position() -> None:
    recs = [np.array([5, 1, 2]), np.array([3, 4, 0]), np.array([], dtype=np.int64)]
    targets = np.array([5, 0, 1])
    counts = np.array([1, 1, 1, 1, 1, 1])
    m = compute_metrics(recs, targets, counts, n_items=6, k=3)
    assert m.hit_rate == pytest.approx(2 / 3)
    assert m.ndcg == pytest.approx((1.0 + 1.0 / np.log2(4)) / 3)
    assert m.coverage == pytest.approx(6 / 6)
    assert m.users == 3


def test_popularity_bias_measures() -> None:
    counts = np.array([100, 50, 1, 1, 1, 1, 1, 1, 1, 1])
    assert head_items(counts).tolist() == [0, 1]
    m = compute_metrics([np.array([0, 2])], np.array([9]), counts, n_items=10)
    assert m.avg_popularity == pytest.approx(50.5)
    assert m.long_tail_share == pytest.approx(0.5)


def test_bootstrap_interval_brackets_the_mean() -> None:
    values = np.array([0.0] * 90 + [1.0] * 10)
    low, high = bootstrap_ci(values)
    assert low < 0.1 < high
    assert bootstrap_ci(values) == (low, high)  # seeded


def test_evaluate_never_recommends_training_items(ratings: pd.DataFrame, catalog: Catalog) -> None:
    split = leave_last_out(ratings)
    data = EvalData.build(split.train, split.test, catalog)
    n = len(catalog)
    everything = from_row_scores(n, n, ((r, np.full(n, 0.5)) for r in range(n)))
    scorer = Scorer(
        everything, everything, collab.item_counts(split.train, catalog), HybridParams(1, 1)
    )
    query = data.profile(catalog, None)
    scores = (query @ scorer.item_matrix("content")).toarray()
    for u in range(query.shape[0]):
        seen = set(data.seen.indices[data.seen.indptr[u] : data.seen.indptr[u + 1]].tolist())
        recs = scorer.rank(np.array(scores[u]), np.array(sorted(seen), dtype=np.int64), 10)
        assert not seen & set(recs.tolist())
    assert evaluate("popularity", scorer, query, data).users == 4


def test_recent_profile_keeps_latest_items(ratings: pd.DataFrame, catalog: Catalog) -> None:
    split = leave_last_out(ratings)
    data = EvalData.build(split.train, split.test, catalog)
    recent = data.profile(catalog, 1)
    user1 = recent.indices[recent.indptr[0] : recent.indptr[1]].tolist()
    assert user1 == [catalog.row_of(50)]
