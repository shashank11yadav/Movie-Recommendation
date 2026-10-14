"""Turning neighbour lists into ranked, explained recommendations.

The browser (web/src/lib/recommend.ts) implements the same arithmetic. A golden
file exported by the pipeline is replayed by the frontend tests to keep the two
in step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as sp

from movierec import config
from movierec.catalog import Catalog
from movierec.neighbors import FloatArray, IntArray, Neighbors

Model = Literal["popularity", "content", "collab", "hybrid"]
MODELS: tuple[Model, ...] = ("popularity", "content", "collab", "hybrid")
# Scores are rounded before ranking so float summation order (sparse matmul here,
# a plain loop in the browser) cannot reorder near-equal candidates.
SCORE_DECIMALS = 9


@dataclass(frozen=True)
class HybridParams:
    """Per liked movie h with n_h ratings: weight w = n_h / (n_h + beta) on the
    collaborative list and (1 - w) * gamma on the content list. Movies with few
    ratings lean on content, well-rated ones on co-ratings."""

    beta: float
    gamma: float


def source_weights(
    model: Model, counts: IntArray, params: HybridParams
) -> tuple[FloatArray, FloatArray]:
    """(collab weight, content weight) for each catalog row used as a liked movie."""
    n = counts.astype(np.float64)
    if model == "content":
        return np.zeros_like(n), np.ones_like(n)
    if model == "collab":
        return np.ones_like(n), np.zeros_like(n)
    if model == "hybrid":
        w = n / (n + params.beta)
        return w, (1.0 - w) * params.gamma
    raise ValueError(f"{model} has no neighbour lists")


@dataclass(frozen=True)
class Scorer:
    content: Neighbors
    collab: Neighbors
    counts: IntArray
    params: HybridParams

    def item_matrix(self, model: Model) -> sp.csr_array:
        """Item x item matrix M so that a liked-set indicator vector v scores as v @ M."""
        w_cf, w_ct = source_weights(model, self.counts, self.params)
        blended = (
            sp.diags_array(w_cf) @ self.collab.to_sparse()
            + sp.diags_array(w_ct) @ self.content.to_sparse()
        )
        return sp.csr_array(blended)

    def rank(self, scores: FloatArray, exclude: IntArray, n: int) -> IntArray:
        """Top-n rows with positive score, ordered by score, then popularity, then row."""
        rounded = np.round(scores, SCORE_DECIMALS)
        rounded[exclude] = 0.0
        candidates = np.flatnonzero(rounded > 0)
        order = np.lexsort((candidates, -self.counts[candidates], -rounded[candidates]))
        return candidates[order[:n]].astype(np.int64)


@dataclass(frozen=True)
class Recommendation:
    row: int
    score: float
    collab_share: float
    because: int
    shared_genres: list[str] = field(default_factory=list)
    shared_tags: list[str] = field(default_factory=list)


def recommend(
    scorer: Scorer,
    catalog: Catalog,
    liked: Sequence[int],
    model: Model,
    n: int = config.EVAL_CUTOFF,
) -> list[Recommendation]:
    """Blend the neighbour lists of every liked row and explain each result."""
    if model == "popularity":
        raise ValueError("the popularity baseline is evaluation-only")
    liked_rows = list(dict.fromkeys(int(r) for r in liked))
    w_cf, w_ct = source_weights(model, scorer.counts, scorer.params)

    from_cf = np.zeros(len(catalog), dtype=np.float64)
    per_source = np.zeros((len(liked_rows), len(catalog)), dtype=np.float64)
    for i, src in enumerate(liked_rows):
        for nb, s in scorer.collab.of(src):
            part = w_cf[src] * s / config.SCORE_SCALE
            from_cf[nb] += part
            per_source[i, nb] += part
        for nb, s in scorer.content.of(src):
            per_source[i, nb] += w_ct[src] * s / config.SCORE_SCALE
    total = per_source.sum(axis=0)

    top = scorer.rank(total, np.asarray(liked_rows, dtype=np.int64), n)
    results = []
    for row in top.tolist():
        because = liked_rows[int(np.argmax(per_source[:, row]))]
        results.append(
            Recommendation(
                row=row,
                score=round(float(total[row]), SCORE_DECIMALS),
                collab_share=float(from_cf[row] / total[row]),
                because=because,
                shared_genres=sorted(set(catalog.genres[row]) & set(catalog.genres[because])),
                shared_tags=[t for t in catalog.tags[row] if t in set(catalog.tags[because])],
            )
        )
    return results
