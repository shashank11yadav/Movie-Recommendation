"""Download, verify and load MovieLens ml-latest-small.

Raw data is never committed. It is fetched from GroupLens over HTTPS, checked
against a pinned SHA-256 and extracted member by member, so a tampered or
truncated archive fails loudly instead of producing a quietly wrong model.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from movierec import config


class DatasetError(RuntimeError):
    """The archive or its contents are not what the pipeline expects."""


@dataclass(frozen=True)
class Dataset:
    movies: pd.DataFrame
    ratings: pd.DataFrame
    tags: pd.DataFrame


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, dest: Path, max_bytes: int) -> None:
    if not url.startswith("https://"):
        raise DatasetError(f"refusing non-HTTPS dataset URL: {url}")
    # Scheme is checked above, so file:// and custom handlers cannot be reached.
    request = urllib.request.Request(url, headers={"User-Agent": "movierec-pipeline"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        written = 0
        with dest.open("wb") as out:
            while chunk := response.read(1 << 16):
                written += len(chunk)
                if written > max_bytes:
                    raise DatasetError(f"download exceeded {max_bytes} bytes; aborting")
                out.write(chunk)


def download(
    data_dir: Path,
    url: str = config.DATASET_URL,
    expected_sha256: str = config.DATASET_SHA256,
) -> Path:
    """Fetch the archive into ``data_dir`` unless a verified copy is already there."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "ml-latest-small.zip"
    if archive.exists() and sha256_of(archive) == expected_sha256:
        return archive

    with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _fetch(url, tmp_path, config.DATASET_MAX_BYTES)
        actual = sha256_of(tmp_path)
        if actual != expected_sha256:
            raise DatasetError(f"checksum mismatch: expected {expected_sha256}, got {actual}")
        tmp_path.replace(archive)
    finally:
        tmp_path.unlink(missing_ok=True)
    return archive


def extract(archive: Path, data_dir: Path) -> Path:
    """Extract only the known CSVs by exact name, which rules out zip-slip paths."""
    out_dir = data_dir / config.DATASET_ARCHIVE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for member in config.DATASET_MEMBERS:
            inner = f"{config.DATASET_ARCHIVE_DIR}/{member}"
            if inner not in names:
                raise DatasetError(f"archive is missing {inner}")
            with zf.open(inner) as src, (out_dir / member).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return out_dir


def ensure_dataset(data_dir: Path) -> Path:
    archive = download(data_dir)
    return extract(archive, data_dir)


def load(csv_dir: Path) -> Dataset:
    """Read the CSVs with explicit dtypes and basic sanity checks."""
    movies = pd.read_csv(
        csv_dir / "movies.csv",
        dtype={"movieId": "int64", "title": "string", "genres": "string"},
        keep_default_na=False,
    )
    ratings = pd.read_csv(
        csv_dir / "ratings.csv",
        dtype={"userId": "int64", "movieId": "int64", "rating": "float64", "timestamp": "int64"},
    )
    tags = pd.read_csv(
        csv_dir / "tags.csv",
        dtype={"userId": "int64", "movieId": "int64", "tag": "string", "timestamp": "int64"},
        keep_default_na=False,
    )

    if movies["movieId"].duplicated().any():
        raise DatasetError("movies.csv has duplicate movieId values")
    known = set(movies["movieId"].tolist())
    if not set(ratings["movieId"].unique().tolist()) <= known:
        raise DatasetError("ratings.csv references movies missing from movies.csv")
    if ratings.duplicated(["userId", "movieId"]).any():
        raise DatasetError("ratings.csv has repeated (userId, movieId) pairs")
    if not ratings["rating"].between(0.5, 5.0).all():
        raise DatasetError("ratings outside the 0.5-5.0 scale")

    return Dataset(movies=movies, ratings=ratings, tags=tags[tags["movieId"].isin(known)])
