"""Shared fixtures: a tiny hand-made catalog and ratings table."""

from __future__ import annotations

import pandas as pd
import pytest

from movierec.catalog import Catalog


@pytest.fixture
def movies() -> pd.DataFrame:
    # Deliberately unsorted with gaps in the ids, the situation that broke the
    # old label-vs-position lookup.
    return pd.DataFrame(
        {
            "movieId": [50, 3, 296, 10, 7],
            "title": [
                "Usual Suspects, The (1995)",
                "Grumpier Old Men (1995)",
                "Pulp Fiction (1994)",
                "GoldenEye (1995)",
                "Muppet Treasure Island (1996)",
            ],
            "genres": [
                "Crime|Mystery|Thriller",
                "Comedy|Romance",
                "Comedy|Crime|Drama|Thriller",
                "Action|Adventure|Thriller",
                "Adventure|Children|Comedy|Musical",
            ],
        }
    )


@pytest.fixture
def tags() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 2, 3, 1],
            "movieId": [296, 296, 50, 7],
            "tag": ["Quentin Tarantino", "quentin tarantino ", "twist ending", "muppets"],
            "timestamp": [10, 11, 12, 13],
        }
    )


@pytest.fixture
def catalog(movies: pd.DataFrame, tags: pd.DataFrame) -> Catalog:
    return Catalog.build(movies, tags)


@pytest.fixture
def ratings() -> pd.DataFrame:
    rows = [
        # user, movie, rating, timestamp
        (1, 296, 5.0, 100),
        (1, 50, 4.5, 101),
        (1, 10, 3.0, 102),
        (2, 296, 4.0, 200),
        (2, 50, 4.0, 201),
        (2, 3, 2.0, 202),
        (3, 296, 5.0, 300),
        (3, 50, 5.0, 300),
        (3, 7, 1.0, 301),
        (4, 3, 3.0, 400),
        (4, 7, 4.0, 401),
        (4, 10, 4.0, 402),
    ]
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
