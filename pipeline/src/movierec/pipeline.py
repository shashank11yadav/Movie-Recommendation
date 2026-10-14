"""End-to-end run: tune on a validation split, report on the test split, refit on everything.

Three catalogs are built from the same movie list (so row indices agree) but
with different tag sets. A user's tag on a movie they later rate is a hint
about that rating, so tags attached to a held-out (user, movie) pair are dropped
before building the content model used to score that split.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from movierec import collab, config, content
from movierec.catalog import Catalog
from movierec.content import Embeddings
from movierec.data import Dataset
from movierec.evaluate import EvalData, Metrics, evaluate
from movierec.neighbors import Neighbors
from movierec.scoring import MODELS, HybridParams, Model, Scorer
from movierec.split import Split, leave_last_out

log = logging.getLogger(__name__)

# Query settings. "recent-5" mirrors the demo, where someone picks a handful of
# films; "full" uses every training rating, which is the usual offline setup.
PROFILES: dict[str, int | None] = {"recent-5": 5, "full": None}
TUNING_PROFILE = "recent-5"

K_GRID = (10, 15, 20, 30)
SHRINK_GRID = (0.0, 5.0, 10.0, 25.0, 50.0)
BETA_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)
GAMMA_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
# Placeholder used while tuning single models; hybrid params do not affect them.
_NEUTRAL = HybridParams(beta=10.0, gamma=1.0)


@dataclass(frozen=True)
class Tuned:
    embedding: str
    k: int
    shrink: float
    hybrid: HybridParams
    trials: list[dict[str, Any]]


@dataclass(frozen=True)
class Fitted:
    catalog: Catalog
    scorer: Scorer


def drop_tags_for(tags: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    held = pd.MultiIndex.from_frame(pairs[["userId", "movieId"]])
    keys = pd.MultiIndex.from_frame(tags[["userId", "movieId"]])
    return tags[~keys.isin(held)]


def _fit(
    ds: Dataset,
    ratings: pd.DataFrame,
    tags: pd.DataFrame,
    embedding: str,
    k: int,
    shrink: float,
    params: HybridParams,
    cache_dir: Path,
) -> Fitted:
    catalog = Catalog.build(ds.movies, tags)
    vectors = content.embed(content.documents(catalog), embedding, cache_dir)
    scorer = Scorer(
        content=content.neighbors(vectors, k=k),
        collab=collab.neighbors(ratings, catalog, k=k, shrink=shrink),
        counts=collab.item_counts(ratings, catalog),
        params=params,
    )
    return Fitted(catalog, scorer)


def _score(
    fitted: Fitted, split: Split, profiles: dict[str, int | None]
) -> dict[str, dict[str, Metrics]]:
    data = EvalData.build(split.train, split.test, fitted.catalog)
    out: dict[str, dict[str, Metrics]] = {}
    for name, size in profiles.items():
        query = data.profile(fitted.catalog, size)
        out[name] = {m: evaluate(m, fitted.scorer, query, data) for m in MODELS}
    return out


def tune(ds: Dataset, outer: Split, cache_dir: Path) -> Tuned:
    """Grid search on the validation split only (each user's second-to-last rating).

    The embedding model is picked first at the default list length. Then, for
    every list length K, the shrinkage is picked on item-kNN alone and the
    hybrid weights on top of it. The winning K is the one with the best hybrid.
    """
    inner = leave_last_out(outer.train)
    tags = drop_tags_for(ds.tags, pd.concat([outer.test, inner.test]))
    catalog = Catalog.build(ds.movies, tags)
    data = EvalData.build(inner.train, inner.test, catalog)
    query = data.profile(catalog, PROFILES[TUNING_PROFILE])
    counts = collab.item_counts(inner.train, catalog)
    empty = collab.neighbors(inner.train.iloc[:0], catalog)
    trials: list[dict[str, Any]] = []

    def record(stage: str, model: Model, scorer: Scorer, **params: object) -> float:
        ndcg = evaluate(model, scorer, query, data).ndcg
        trials.append({"stage": stage, **params, "ndcg": round(ndcg, 4)})
        return ndcg

    vectors: dict[str, Embeddings] = {}
    by_embedding: dict[str, float] = {}
    for key in config.EMBEDDING_MODELS:
        vectors[key] = content.embed(content.documents(catalog), key, cache_dir)
        lists = content.neighbors(vectors[key])
        scorer = Scorer(lists, empty, counts, _NEUTRAL)
        by_embedding[key] = record("embedding", "content", scorer, embedding=key)
    embedding = max(by_embedding, key=by_embedding.__getitem__)

    best: tuple[float, int, float, HybridParams] | None = None
    for k in K_GRID:
        content_lists = content.neighbors(vectors[embedding], k=k)
        by_shrink: dict[float, tuple[float, Neighbors]] = {}
        for shrink in SHRINK_GRID:
            lists = collab.neighbors(inner.train, catalog, k=k, shrink=shrink)
            scorer = Scorer(empty, lists, counts, _NEUTRAL)
            by_shrink[shrink] = (record("shrink", "collab", scorer, k=k, shrink=shrink), lists)
        shrink = max(by_shrink, key=lambda s: by_shrink[s][0])
        for beta, gamma in itertools.product(BETA_GRID, GAMMA_GRID):
            params = HybridParams(beta=beta, gamma=gamma)
            scorer = Scorer(content_lists, by_shrink[shrink][1], counts, params)
            ndcg = record("hybrid", "hybrid", scorer, k=k, shrink=shrink, beta=beta, gamma=gamma)
            if best is None or ndcg > best[0]:
                best = (ndcg, k, shrink, params)
    if best is None:
        raise RuntimeError("tuning grid is empty")
    _, k, shrink, params = best
    log.info("tuned: embedding=%s k=%s shrink=%s hybrid=%s", embedding, k, shrink, params)
    return Tuned(embedding=embedding, k=k, shrink=shrink, hybrid=params, trials=trials)


def run(ds: Dataset, cache_dir: Path) -> tuple[dict[str, Any], Fitted, Tuned]:
    """Returns (results document, model fitted on all data, tuned settings)."""
    outer = leave_last_out(ds.ratings)
    tuned = tune(ds, outer, cache_dir)

    test_tags = drop_tags_for(ds.tags, outer.test)
    fitted_test = _fit(
        ds, outer.train, test_tags, tuned.embedding, tuned.k, tuned.shrink, tuned.hybrid, cache_dir
    )
    scores = _score(fitted_test, outer, PROFILES)

    full = _fit(
        ds, ds.ratings, ds.tags, tuned.embedding, tuned.k, tuned.shrink, tuned.hybrid, cache_dir
    )
    query_coverage: dict[str, float] = {}
    for model in ("content", "collab", "hybrid"):
        per_movie = np.diff(full.scorer.item_matrix(model).indptr)
        query_coverage[model] = round(float((per_movie >= config.EVAL_CUTOFF).mean()), 4)
    results = {
        "dataset": "MovieLens ml-latest-small",
        "dataset_sha256": config.DATASET_SHA256,
        "split": "temporal leave-last-out per user; hyperparameters tuned on the "
        "second-to-last rating, metrics on the last",
        "users": int(outer.test["userId"].nunique()),
        "catalog_size": len(full.catalog),
        "k": config.EVAL_CUTOFF,
        "neighbors_per_movie": tuned.k,
        "settings": {
            "neighbors_per_movie": tuned.k,
            "embedding_key": tuned.embedding,
            "embedding_model": config.EMBEDDING_MODELS[tuned.embedding].name,
            "embedding_revision": config.EMBEDDING_MODELS[tuned.embedding].revision,
            "cf_shrink": tuned.shrink,
            "cf_min_ratings": config.MIN_RATINGS_FOR_CF,
            "hybrid_beta": tuned.hybrid.beta,
            "hybrid_gamma": tuned.hybrid.gamma,
            "tuning_profile": TUNING_PROFILE,
        },
        "profiles": {
            profile: {model: m.as_dict() for model, m in by_model.items()}
            for profile, by_model in scores.items()
        },
        # Share of movies that, picked on their own in the demo, return a full
        # list of recommendations. Collaborative lists need ratings; content does not.
        "query_coverage": query_coverage,
        "tuning": tuned.trials,
    }
    return results, full, tuned


def fit_serving(ds: Dataset, cache_dir: Path, settings: dict[str, Any]) -> Fitted:
    """Refit on all data with settings from a previous ``run`` (as stored in results.json)."""
    params = HybridParams(
        beta=float(settings["hybrid_beta"]), gamma=float(settings["hybrid_gamma"])
    )
    return _fit(
        ds,
        ds.ratings,
        ds.tags,
        str(settings["embedding_key"]),
        int(settings["neighbors_per_movie"]),
        float(settings["cf_shrink"]),
        params,
        cache_dir,
    )
