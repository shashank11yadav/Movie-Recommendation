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
# files.grouplens.org serves a chain that ends at "emSign Root TLS CA - G3", which is
# not in Python's trust store (certifi) or the usual system stores, so every client
# fails to verify it. This CA, as the server sends it and as logged on crt.sh, is
# trusted for the dataset download only; the archive is still checked against
# DATASET_SHA256. Drop it once the chain verifies without it.
GROUPLENS_CA_SHA256 = "3094025792039a68af6e553686410bdab7c0d4a36b9084cba64202e35d6ebbdd"
GROUPLENS_CA_PEM = """\
-----BEGIN CERTIFICATE-----
MIIDdDCCAvmgAwIBAgIQAJYaPvsTWVdu0EAFtGhKWzAKBggqhkjOPQQDAzBrMQsw
CQYDVQQGEwJJTjETMBEGA1UECxMKZW1TaWduIFBLSTElMCMGA1UEChMcZU11ZGhy
YSBUZWNobm9sb2dpZXMgTGltaXRlZDEgMB4GA1UEAxMXZW1TaWduIEVDQyBSb290
IENBIC0gRzMwHhcNMjQwNzExMTU0MzM2WhcNMzkwNzA5MTgzMDAwWjBWMQswCQYD
VQQGEwJJTjElMCMGA1UECgwcZU11ZGhyYSBUZWNobm9sb2dpZXMgTGltaXRlZDEg
MB4GA1UEAwwXZW1TaWduIFJvb3QgVExTIENBIC0gRzMwdjAQBgcqhkjOPQIBBgUr
gQQAIgNiAARfD9w/q1cWxL6D9bMGH6CQ4u+gAV6GN9s+PwQx9Bl4hC+0iU8QOk8m
WO/RwXQy10DCSxT8vSIJClHKp+Ao4qJKdYeV0SwZ7LMQziyLfxIUOY6Grja/sQFw
LKYf8EdDXAyjggF1MIIBcTAfBgNVHSMEGDAWgBR8XQKEE9TMipuBzhccLikenEhj
QjAdBgNVHQ4EFgQU4TqrIXzx401UrtRyt/2LWUIgkFQwHQYDVR0lBBYwFAYIKwYB
BQUHAwEGCCsGAQUFBwMCMD0GA1UdIAQ2MDQwMgYEVR0gADAqMCgGCCsGAQUFBwIB
FhxodHRwOi8vcmVwb3NpdG9yeS5lbXNpZ24uY29tMHgGCCsGAQUFBwEBBGwwajAi
BggrBgEFBQcwAYYWaHR0cDovL29jc3AuZW1zaWduLmNvbTBEBggrBgEFBQcwAoY4
aHR0cDovL3JlcG9zaXRvcnkuZW1zaWduLmNvbS9jZXJ0cy9lbVNpZ25FQ0NSb290
Q0FHMy5jcnQwMwYDVR0fBCwwKjAooCagJIYiaHR0cDovL2NybC5lbXNpZ24uY29t
P1Jvb3RDQUczLmNybDAOBgNVHQ8BAf8EBAMCAQYwEgYDVR0TAQH/BAgwBgEB/wIB
ATAKBggqhkjOPQQDAwNpADBmAjEAt9CIpRGp46QHQYDUBnLKw+vvK0/h9vmilEAg
TroUl/anH4x2rXHqiCtBTBfgKDegAjEA8WpG1LTZdAIL0FSYqxXeREaeTSxEofM0
1jJH3jGvCXC79LU9UPJ2o3TU+UIj2gMV
-----END CERTIFICATE-----
"""


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
