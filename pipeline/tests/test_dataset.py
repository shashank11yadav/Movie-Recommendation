"""Tests against the real MovieLens files. Run `uv run movierec download` first."""

from __future__ import annotations

from pathlib import Path

import pytest

from movierec import config, data
from movierec.catalog import Catalog
from movierec.split import leave_last_out

CSV_DIR = config.DEFAULT_DATA_DIR / config.DATASET_ARCHIVE_DIR

pytestmark = [
    pytest.mark.dataset,
    pytest.mark.skipif(not (CSV_DIR / "ratings.csv").exists(), reason="dataset not downloaded"),
]


@pytest.fixture(scope="module")
def dataset() -> data.Dataset:
    return data.load(Path(CSV_DIR))


def test_archive_checksum_is_pinned() -> None:
    archive = config.DEFAULT_DATA_DIR / "ml-latest-small.zip"
    assert data.sha256_of(archive) == config.DATASET_SHA256


def test_every_id_maps_to_its_own_row(dataset: data.Dataset) -> None:
    catalog = Catalog.build(dataset.movies, dataset.tags)
    assert len(catalog) == 9742
    for row, movie_id in enumerate(catalog.ids.tolist()):
        assert catalog.row_of(movie_id) == row
    # The last three titles crashed the old app with IndexError.
    for movie_id in dataset.movies["movieId"].tail(3):
        assert catalog.ids[catalog.row_of(int(movie_id))] == movie_id


def test_real_split_has_no_leakage(dataset: data.Dataset) -> None:
    split = leave_last_out(dataset.ratings)
    assert split.test["userId"].nunique() == 610 == len(split.test)
    merged = split.train.merge(split.test, on="userId", suffixes=("", "_test"))
    assert (merged["timestamp"] <= merged["timestamp_test"]).all()
    assert not (merged["movieId"] == merged["movieId_test"]).any()
