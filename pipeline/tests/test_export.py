"""Checks on the committed web export: the files the browser actually loads."""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import pytest

from movierec import config

MAX_EXPORT_BYTES = 3 * 1024 * 1024
PULP_FICTION = 296


@cache
def movies() -> dict[str, Any]:
    return json.loads((config.WEB_DATA_DIR / "movies.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@cache
def neighbors() -> dict[str, Any]:
    return json.loads((config.WEB_DATA_DIR / "neighbors.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def lists(model: str) -> list[list[tuple[int, int]]]:
    flat: list[list[int]] = neighbors()[model]
    return [list(zip(row[::2], row[1::2], strict=True)) for row in flat]


def test_export_fits_size_budget() -> None:
    total = sum(p.stat().st_size for p in config.WEB_DATA_DIR.glob("*.json"))
    assert total < MAX_EXPORT_BYTES


def test_columns_line_up() -> None:
    m = movies()
    n = len(m["id"])
    for column in ("title", "year", "genre", "tags", "ratings"):
        assert len(m[column]) == n, column
    assert len(set(m["id"])) == n
    assert m["id"] == sorted(m["id"])
    assert len(neighbors()["content"]) == len(neighbors()["collab"]) == n


@pytest.mark.parametrize("model", ["content", "collab"])
def test_every_list_is_valid(model: str) -> None:
    n = len(movies()["id"])
    for row, pairs in enumerate(lists(model)):
        rows = [r for r, _ in pairs]
        assert row not in rows, f"movie at row {row} lists itself"
        assert len(rows) == len(set(rows)), f"duplicate neighbour at row {row}"
        assert all(0 <= r < n for r in rows)
        assert all(0 < s <= neighbors()["scale"] for _, s in pairs)
        assert len(pairs) <= int(json.loads(config.RESULTS_PATH.read_text())["neighbors_per_movie"])


def test_every_movie_has_content_neighbours() -> None:
    assert all(pairs for pairs in lists("content"))


@pytest.mark.parametrize("model", ["content", "collab"])
def test_pulp_fiction_neighbours_are_related(model: str) -> None:
    """Regression for the old label/position bug that returned Muppets films."""
    m = movies()
    row = m["id"].index(PULP_FICTION)
    assert m["title"][row] == "Pulp Fiction"
    pulp_genres = set(m["genre"][row])
    pairs = lists(model)[row]
    assert len(pairs) >= 5
    for nb, _ in pairs:
        assert "muppet" not in m["title"][nb].lower()
    sharing = [nb for nb, _ in pairs if pulp_genres & set(m["genre"][nb])]
    # Content neighbours come from genre/tag text, so all should overlap.
    # Co-rating neighbours need not, but most still do.
    assert len(sharing) >= (len(pairs) if model == "content" else len(pairs) // 2)


def test_known_film_has_expected_neighbour() -> None:
    m = movies()
    pulp = m["id"].index(PULP_FICTION)
    reservoir = m["id"].index(1089)  # Reservoir Dogs
    assert reservoir in {nb for nb, _ in lists("content")[pulp]}


def test_hybrid_settings_match_results() -> None:
    settings = json.loads(config.RESULTS_PATH.read_text())["settings"]
    assert neighbors()["hybrid"] == {
        "beta": settings["hybrid_beta"],
        "gamma": settings["hybrid_gamma"],
    }


def test_golden_cases_agree_with_export() -> None:
    """Recompute the golden cases straight from the JSON to catch a stale export."""
    golden = json.loads(config.GOLDEN_PATH.read_text(encoding="utf-8"))
    m, nb = movies(), neighbors()
    row_of = {movie_id: row for row, movie_id in enumerate(m["id"])}
    beta, gamma, scale = nb["hybrid"]["beta"], nb["hybrid"]["gamma"], nb["scale"]
    for case in golden["cases"]:
        scores: dict[int, float] = {}
        for movie_id in case["liked"]:
            src = row_of[movie_id]
            n = m["ratings"][src] if lists("collab")[src] else 0
            w_cf = {"content": 0.0, "collab": 1.0}.get(case["model"], n / (n + beta))
            w_ct = {"content": 1.0, "collab": 0.0}.get(case["model"], (1 - w_cf) * gamma)
            for model, weight in (("collab", w_cf), ("content", w_ct)):
                for r, s in lists(model)[src]:
                    scores[r] = scores.get(r, 0.0) + weight * s / scale
        for result in case["results"]:
            assert scores[row_of[result["id"]]] == pytest.approx(result["score"], abs=1e-9)
