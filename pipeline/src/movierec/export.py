"""Write the static files the web app loads, plus a golden file for parity tests.

Neighbour references are catalog row indices into the ``movies`` arrays of the
same export, which keeps the files small. The parity test in
``tests/test_export.py`` checks every reference against movie ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from movierec import config
from movierec.catalog import Catalog
from movierec.neighbors import Neighbors
from movierec.scoring import Model, Scorer, recommend

DATA_NOTICE = (
    "Derived from MovieLens ml-latest-small (GroupLens Research, University of Minnesota). "
    "Redistributed under the MovieLens usage license: research and non-commercial use only, "
    "no endorsement implied. See README Credits. Not covered by the repository MIT license."
)

# Liked-movie sets replayed by the frontend tests. A well-known film, a pair, and
# a movie with too few ratings for collaborative neighbours.
GOLDEN_QUERIES: tuple[tuple[int, ...], ...] = ((296,), (1, 6377), (2571, 318, 858), (40,))


def _dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")


def _pairs(neighbors: Neighbors) -> list[list[int]]:
    """Each row as a flat [row, score, row, score, ...] list."""
    return [[v for pair in neighbors.of(row) for v in pair] for row in range(neighbors.n_items)]


def movies_payload(catalog: Catalog, scorer: Scorer) -> dict[str, Any]:
    genre_names = sorted({g for gs in catalog.genres for g in gs})
    genre_index = {g: i for i, g in enumerate(genre_names)}
    return {
        "notice": DATA_NOTICE,
        "genres": genre_names,
        "id": catalog.ids.tolist(),
        "title": catalog.titles,
        "year": catalog.years,
        "genre": [[genre_index[g] for g in gs] for gs in catalog.genres],
        "tags": catalog.tags,
        "ratings": scorer.counts.tolist(),
    }


def neighbors_payload(scorer: Scorer) -> dict[str, Any]:
    return {
        "notice": DATA_NOTICE,
        "scale": config.SCORE_SCALE,
        "hybrid": {"beta": scorer.params.beta, "gamma": scorer.params.gamma},
        "content": _pairs(scorer.content),
        "collab": _pairs(scorer.collab),
    }


def golden_payload(catalog: Catalog, scorer: Scorer) -> dict[str, Any]:
    cases = []
    models: tuple[Model, ...] = ("content", "collab", "hybrid")
    for liked in GOLDEN_QUERIES:
        rows = catalog.rows_of(liked).tolist()
        for model in models:
            recs = recommend(scorer, catalog, rows, model)
            cases.append(
                {
                    "liked": list(liked),
                    "model": model,
                    "results": [
                        {
                            "id": int(catalog.ids[r.row]),
                            "score": r.score,
                            "collabShare": round(r.collab_share, 6),
                            "because": int(catalog.ids[r.because]),
                            "sharedGenres": r.shared_genres,
                            "sharedTags": r.shared_tags,
                        }
                        for r in recs
                    ],
                }
            )
    return {"cases": cases}


def metrics_payload(results: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "users",
        "catalog_size",
        "k",
        "neighbors_per_movie",
        "settings",
        "profiles",
        "query_coverage",
    )
    return {key: results[key] for key in keep}


def write_all(
    catalog: Catalog,
    scorer: Scorer,
    results: dict[str, Any],
    web_dir: Path = config.WEB_DATA_DIR,
    results_path: Path = config.RESULTS_PATH,
    golden_path: Path = config.GOLDEN_PATH,
) -> list[Path]:
    written = {
        web_dir / "movies.json": movies_payload(catalog, scorer),
        web_dir / "neighbors.json": neighbors_payload(scorer),
        web_dir / "metrics.json": metrics_payload(results),
        golden_path: golden_payload(catalog, scorer),
    }
    for path, payload in written.items():
        _dump(path, payload)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return [*written, results_path]
