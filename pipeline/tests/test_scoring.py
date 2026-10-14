from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from movierec import collab
from movierec.catalog import Catalog
from movierec.neighbors import from_row_scores
from movierec.scoring import HybridParams, Scorer, recommend, source_weights


@pytest.fixture
def scorer(ratings: pd.DataFrame, catalog: Catalog) -> Scorer:
    n = len(catalog)
    # Content: everything mildly similar to everything, Muppets close to GoldenEye.
    sims = np.full((n, n), 0.3)
    sims[catalog.row_of(7), catalog.row_of(10)] = 0.9
    content_nb = from_row_scores(n, 3, ((r, sims[r]) for r in range(n)))
    return Scorer(
        content=content_nb,
        collab=collab.neighbors(ratings, catalog, k=3, min_ratings=3),
        counts=collab.item_counts(ratings, catalog),
        params=HybridParams(beta=2.0, gamma=1.0),
    )


def test_hybrid_weights_shift_to_content_for_cold_items() -> None:
    w_cf, w_ct = source_weights("hybrid", np.array([0, 2, 98]), HybridParams(beta=2.0, gamma=0.5))
    assert w_cf.tolist() == [0.0, 0.5, 0.98]
    assert w_ct.tolist() == pytest.approx([0.5, 0.25, 0.01])


@pytest.mark.parametrize("model", ["content", "collab", "hybrid"])
def test_recommendations_exclude_liked_and_have_no_duplicates(
    scorer: Scorer, catalog: Catalog, model: str
) -> None:
    liked = [catalog.row_of(296), catalog.row_of(50)]
    recs = recommend(scorer, catalog, liked, model)  # type: ignore[arg-type]
    rows = [r.row for r in recs]
    assert len(rows) == len(set(rows))
    assert not set(rows) & set(liked)
    assert all(r.because in liked for r in recs)
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_collab_share_matches_model(scorer: Scorer, catalog: Catalog) -> None:
    liked = [catalog.row_of(296)]
    assert all(r.collab_share == 0.0 for r in recommend(scorer, catalog, liked, "content"))
    assert all(r.collab_share == 1.0 for r in recommend(scorer, catalog, liked, "collab"))


def test_blend_counts_zero_out_movies_without_collab_lists(
    scorer: Scorer, catalog: Catalog
) -> None:
    assert scorer.blend_counts[catalog.row_of(7)] == 0
    assert scorer.blend_counts[catalog.row_of(296)] == 3


def test_cold_movie_falls_back_to_content(scorer: Scorer, catalog: Catalog) -> None:
    muppets = catalog.row_of(7)  # two ratings, below the collaborative threshold
    assert recommend(scorer, catalog, [muppets], "collab") == []
    hybrid = recommend(scorer, catalog, [muppets], "hybrid")
    assert hybrid[0].row == catalog.row_of(10)
    assert hybrid[0].collab_share == 0.0
    assert hybrid[0].shared_genres == ["Adventure"]


def test_duplicate_likes_count_once(scorer: Scorer, catalog: Catalog) -> None:
    once = recommend(scorer, catalog, [catalog.row_of(296)], "hybrid")
    twice = recommend(scorer, catalog, [catalog.row_of(296)] * 2, "hybrid")
    assert once == twice


def test_popularity_is_not_servable(scorer: Scorer, catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="evaluation-only"):
        recommend(scorer, catalog, [0], "popularity")


def test_item_matrix_matches_recommend(scorer: Scorer, catalog: Catalog) -> None:
    liked = [catalog.row_of(296), catalog.row_of(3)]
    vector = np.zeros(len(catalog))
    vector[liked] = 1.0
    batch = vector @ scorer.item_matrix("hybrid").toarray()
    for rec in recommend(scorer, catalog, liked, "hybrid"):
        assert batch[rec.row] == pytest.approx(rec.score, abs=1e-9)
