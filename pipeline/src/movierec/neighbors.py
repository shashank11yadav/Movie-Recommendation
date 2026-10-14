"""Fixed-size neighbour lists, the format shared by evaluation and the web export."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from movierec import config

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Neighbors:
    """Top-k neighbours per catalog row.

    ``rows[i]`` holds neighbour row indices for item ``i`` (padded with -1) and
    ``scores[i]`` the matching similarities in fixed point (``SCORE_SCALE``).
    Scores are quantised before evaluation so the metrics are computed on the
    exact numbers the browser receives.
    """

    rows: IntArray
    scores: IntArray

    @property
    def n_items(self) -> int:
        return int(self.rows.shape[0])

    def of(self, row: int) -> list[tuple[int, int]]:
        keep = self.rows[row] >= 0
        return list(
            zip(self.rows[row][keep].tolist(), self.scores[row][keep].tolist(), strict=True)
        )

    def to_sparse(self) -> sp.csr_array:
        mask = self.rows >= 0
        src = np.repeat(np.arange(self.n_items), mask.sum(axis=1))
        data = self.scores[mask].astype(np.float64) / config.SCORE_SCALE
        return sp.csr_array((data, (src, self.rows[mask])), shape=(self.n_items, self.n_items))


def _rank_row(values: FloatArray, k: int) -> tuple[IntArray, FloatArray]:
    """Indices of the k largest positive values, ties broken by lower index."""
    positive = np.flatnonzero(values > 0)
    if positive.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    if positive.size > k:
        # Keep everything tied with the k-th value so the final order stays deterministic.
        kth = np.partition(values[positive], positive.size - k)[positive.size - k]
        positive = positive[values[positive] >= kth]
    order = np.lexsort((positive, -values[positive]))[:k]
    picked = positive[order]
    return picked.astype(np.int64), values[picked]


def _quantise(scores: FloatArray) -> IntArray:
    return np.rint(np.clip(scores, 0.0, 1.0) * config.SCORE_SCALE).astype(np.int64)


def from_row_scores(
    n_items: int,
    k: int,
    row_scores: Iterable[tuple[int, FloatArray]],
) -> Neighbors:
    """Build lists from (row, similarity-to-every-item) pairs. The row itself is excluded."""
    rows = np.full((n_items, k), -1, dtype=np.int64)
    scores = np.zeros((n_items, k), dtype=np.int64)
    for row, values in row_scores:
        values = values.astype(np.float64, copy=True)
        values[row] = 0.0
        idx, vals = _rank_row(values, k)
        q = _quantise(vals)
        keep = q > 0
        n = int(keep.sum())
        rows[row, :n] = idx[keep]
        scores[row, :n] = q[keep]
    return Neighbors(rows=rows, scores=scores)
