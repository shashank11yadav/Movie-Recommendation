"""Item-item collaborative filtering (item-kNN with shrunk cosine) and the popularity baseline."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import scipy.sparse as sp

from movierec import config
from movierec.catalog import Catalog
from movierec.neighbors import FloatArray, IntArray, Neighbors, from_row_scores


def interaction_matrix(
    ratings: pd.DataFrame, catalog: Catalog, users: IntArray | None = None
) -> sp.csr_array:
    """Binary user x item matrix with items on catalog rows.

    Matrix row ``u`` belongs to ``users[u]`` (sorted unique ids by default), so
    callers that pass the same ``users`` get row-aligned matrices.
    """
    user_ids = ratings["userId"].to_numpy()
    if users is None:
        users = np.unique(user_ids)
    user_rows = np.searchsorted(users, user_ids)
    if user_rows.size and (user_rows.max() >= len(users) or (users[user_rows] != user_ids).any()):
        raise ValueError("ratings contain users outside the given user list")
    item_rows = catalog.rows_of(ratings["movieId"].to_numpy())
    data = np.ones(len(ratings), dtype=np.float64)
    return sp.csr_array((data, (user_rows, item_rows)), shape=(len(users), len(catalog)))


def item_counts(ratings: pd.DataFrame, catalog: Catalog) -> IntArray:
    counts = np.zeros(len(catalog), dtype=np.int64)
    rows, n = np.unique(catalog.rows_of(ratings["movieId"].to_numpy()), return_counts=True)
    counts[rows] = n
    return counts


def neighbors(
    ratings: pd.DataFrame,
    catalog: Catalog,
    k: int = config.TOP_K,
    shrink: float = config.CF_SHRINK,
    min_ratings: int = config.MIN_RATINGS_FOR_CF,
) -> Neighbors:
    """Shrunk cosine between item columns: co(i, j) / (sqrt(n_i * n_j) + shrink).

    Items below ``min_ratings`` get no list at all. Their co-occurrence counts
    are too small to mean anything, and the hybrid model falls back to content
    for them instead.
    """
    x = interaction_matrix(ratings, catalog)
    co = (x.T @ x).tocsr()
    counts = item_counts(ratings, catalog).astype(np.float64)
    norms = np.sqrt(counts)

    def rows() -> Iterator[tuple[int, FloatArray]]:
        for row in np.flatnonzero(counts >= min_ratings):
            start, end = co.indptr[row], co.indptr[row + 1]
            cols = co.indices[start:end]
            values = np.zeros(len(catalog), dtype=np.float64)
            values[cols] = co.data[start:end] / (norms[row] * norms[cols] + shrink)
            yield int(row), values

    return from_row_scores(len(catalog), k, rows())
