"""Fixed inputs for the pipeline: dataset location, checksums, model ids and paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
# SHA-256 of the archive as published on 2018-09-26. GroupLens only ships an MD5,
# so this was computed once and pinned here; a mismatch aborts the pipeline.
DATASET_SHA256 = "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"
# The real archive is ~1 MB. Anything far larger is not the file we expect.
DATASET_MAX_BYTES = 8 * 1024 * 1024
DATASET_MEMBERS = ("movies.csv", "ratings.csv", "tags.csv", "links.csv", "README.txt")
DATASET_ARCHIVE_DIR = "ml-latest-small"


@dataclass(frozen=True)
class EmbeddingModel:
    name: str
    # Pinned commit so a re-run months later embeds with exactly the same weights.
    revision: str


EMBEDDING_MODELS: dict[str, EmbeddingModel] = {
    "minilm": EmbeddingModel(
        "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    ),
    "bge-small": EmbeddingModel(
        "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    ),
}

# Default neighbours kept per movie. The pipeline tunes this (see K_GRID in
# pipeline.py); evaluation uses the same truncated lists the browser receives.
TOP_K = 20
EVAL_CUTOFF = 10
# Collaborative neighbours are only exported for movies with at least this many
# training ratings; below it the cosine is dominated by one or two users.
MIN_RATINGS_FOR_CF = 3
# Item-kNN shrinkage: sim = co / (sqrt(n_i * n_j) + SHRINK). Damps similarities
# that rest on a handful of co-ratings.
CF_SHRINK = 10.0
# Fixed-point scale for exported similarity scores (3 decimal places).
SCORE_SCALE = 1000

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PIPELINE_ROOT.parent
DEFAULT_DATA_DIR = PIPELINE_ROOT / "data"
RESULTS_PATH = PIPELINE_ROOT / "eval" / "results.json"
WEB_DATA_DIR = REPO_ROOT / "web" / "public" / "data"
GOLDEN_PATH = REPO_ROOT / "web" / "src" / "lib" / "golden.test-data.json"
