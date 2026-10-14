from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from movierec import collab, config, content
from movierec.catalog import Catalog
from movierec.neighbors import from_row_scores


def test_neighbors_exclude_self_and_break_ties_by_row() -> None:
    values = np.array([0.9, 0.5, 0.5, 0.0, 0.5])
    nb = from_row_scores(5, 2, [(0, values)])
    assert nb.of(0) == [(1, 500), (2, 500)]
    assert nb.of(3) == []


def test_neighbors_drop_non_positive_and_pad() -> None:
    nb = from_row_scores(3, 5, [(1, np.array([-0.2, 1.0, 0.25]))])
    assert nb.of(1) == [(2, 250)]
    assert nb.rows[1].tolist() == [2, -1, -1, -1, -1]


def test_to_sparse_round_trips_scores() -> None:
    nb = from_row_scores(3, 2, [(0, np.array([0.0, 0.4, 0.8]))])
    dense = nb.to_sparse().toarray()
    assert dense[0].tolist() == [0.0, 0.4, 0.8]
    assert dense[1:].sum() == 0


def test_item_knn_shrunk_cosine(ratings: pd.DataFrame, catalog: Catalog) -> None:
    nb = collab.neighbors(ratings, catalog, k=5, shrink=1.0, min_ratings=1)
    pulp, suspects = catalog.row_of(296), catalog.row_of(50)
    # 296 and 50 are each rated by users 1, 2, 3: co = 3, n = 3 and 3.
    expected = round(3 / (np.sqrt(3 * 3) + 1.0) * config.SCORE_SCALE)
    assert dict(nb.of(pulp))[suspects] == expected


def test_item_knn_skips_rarely_rated_items(ratings: pd.DataFrame, catalog: Catalog) -> None:
    nb = collab.neighbors(ratings, catalog, k=5, min_ratings=3)
    assert nb.of(catalog.row_of(3)) == []  # two ratings
    assert nb.of(catalog.row_of(296)) != []


def test_item_counts(ratings: pd.DataFrame, catalog: Catalog) -> None:
    counts = collab.item_counts(ratings, catalog)
    assert counts[catalog.row_of(296)] == 3
    assert counts[catalog.row_of(7)] == 2
    assert counts.sum() == len(ratings)


def test_interaction_matrix_rejects_unknown_users(ratings: pd.DataFrame, catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="outside"):
        collab.interaction_matrix(ratings, catalog, users=np.array([1, 2], dtype=np.int64))


def test_content_document_uses_genres_and_tags(catalog: Catalog) -> None:
    doc = content.document(catalog, catalog.row_of(296))
    assert doc == (
        "Pulp Fiction (1994). Genres: Comedy, Crime, Drama, Thriller. Tags: quentin tarantino."
    )


def test_content_neighbors_on_known_vectors() -> None:
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], dtype=np.float32)
    nb = content.neighbors(vectors, k=3, chunk=2)
    rows = [r for r, _ in nb.of(0)]
    assert rows[0] == 1
    assert 0 not in rows
    assert 3 not in rows  # opposite direction, negative cosine
