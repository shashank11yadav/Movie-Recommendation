from __future__ import annotations

import pandas as pd
import pytest

from movierec.catalog import Catalog, clean_tag, parse_title, top_tags


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Matrix, The (1999)", ("The Matrix", 1999)),
        ("Toy Story (1995)", ("Toy Story", 1995)),
        ("City of Lost Children, The (Cité des enfants perdus, La) (1995)",
         ("The City of Lost Children (Cité des enfants perdus, La)", 1995)),
        ("Divo, Il (2008)", ("Il Divo", 2008)),
        ("Cry, the Beloved Country (1995)", ("Cry, the Beloved Country", 1995)),
        ("Babylon 5", ("Babylon 5", None)),
        ("Fawlty Towers (1975-1979)", ("Fawlty Towers", 1975)),
    ],
)  # fmt: skip
def test_parse_title(raw: str, expected: tuple[str, int | None]) -> None:
    assert parse_title(raw) == expected


def test_clean_tag_normalises_and_drops_noise() -> None:
    assert clean_tag("  Quentin   Tarantino ") == "quentin tarantino"
    assert clean_tag("in netflix queue") is None
    assert clean_tag("!!!") is None
    assert clean_tag("x" * 41) is None


def test_top_tags_orders_by_frequency_then_name() -> None:
    assert top_tags(["b", "a", "B", "c", "a", "a"], limit=2) == ["a", "b"]


def test_row_mapping_follows_ids_not_input_order(catalog: Catalog, movies: pd.DataFrame) -> None:
    # Rows are assigned by sorted id; every lookup must land on the right title.
    assert catalog.ids.tolist() == [3, 7, 10, 50, 296]
    for movie_id, raw_title in zip(movies["movieId"], movies["title"], strict=True):
        row = catalog.row_of(int(movie_id))
        assert int(catalog.ids[row]) == movie_id
        assert catalog.titles[row] == parse_title(str(raw_title))[0]


def test_unknown_id_raises(catalog: Catalog) -> None:
    with pytest.raises(KeyError, match="unknown movieId 999"):
        catalog.row_of(999)


def test_duplicate_ids_rejected(movies: pd.DataFrame, tags: pd.DataFrame) -> None:
    doubled = pd.concat([movies, movies.iloc[:1]])
    with pytest.raises(ValueError, match="unique"):
        Catalog.build(doubled, tags)


def test_tags_are_deduplicated_per_movie(catalog: Catalog) -> None:
    assert catalog.tags[catalog.row_of(296)] == ["quentin tarantino"]
    assert catalog.tags[catalog.row_of(3)] == []


def test_find_is_case_and_accent_insensitive() -> None:
    movies = pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["Amélie (Fabuleux destin d'Amélie Poulain, Le) (2001)", "Amen. (2002)"],
            "genres": ["Comedy|Romance", "Drama"],
        }
    )
    empty = pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
    cat = Catalog.build(movies, empty)
    assert cat.find("AMELIE") == [0]
    assert cat.find("   ") == []
