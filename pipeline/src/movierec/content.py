"""Content-based neighbours from sentence embeddings of title, genres and tags."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt

from movierec import config
from movierec.catalog import Catalog
from movierec.neighbors import FloatArray, Neighbors, from_row_scores

Embeddings = npt.NDArray[np.float32]


def document(catalog: Catalog, row: int) -> str:
    """Text that gets embedded for one movie.

    Title alone carries little meaning for a small encoder, so genres and the
    most-applied user tags do most of the work. Movies without tags still get a
    usable vector from title and genres, which is what makes this the cold-start
    fallback.
    """
    parts = [catalog.label(row) + "."]
    if catalog.genres[row]:
        parts.append("Genres: " + ", ".join(catalog.genres[row]) + ".")
    if catalog.tags[row]:
        parts.append("Tags: " + ", ".join(catalog.tags[row]) + ".")
    return " ".join(parts)


def documents(catalog: Catalog) -> list[str]:
    return [document(catalog, row) for row in range(len(catalog))]


def embed(texts: list[str], model_key: str, cache_dir: Path) -> Embeddings:
    """Encode texts with a pinned sentence-transformers model, cached on disk by content hash."""
    spec = config.EMBEDDING_MODELS[model_key]
    digest = hashlib.sha256(
        "\n".join([spec.name, spec.revision, *texts]).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"embeddings-{model_key}-{digest}.npy"
    if cache_file.exists():
        cached: Embeddings = np.load(cache_file, allow_pickle=False)
        return cached

    # Imported lazily: torch is slow to import and tests that use precomputed
    # vectors should not pay for it.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        spec.name,
        revision=spec.revision,
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    result = np.asarray(vectors, dtype=np.float32)
    np.save(cache_file, result, allow_pickle=False)
    return result


def neighbors(embeddings: Embeddings, k: int = config.TOP_K, chunk: int = 1024) -> Neighbors:
    """Cosine top-k over L2-normalised vectors, computed in row chunks to bound memory."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = (embeddings / np.where(norms == 0, 1.0, norms)).astype(np.float32)

    def rows() -> Iterator[tuple[int, FloatArray]]:
        for start in range(0, len(unit), chunk):
            block = unit[start : start + chunk] @ unit.T
            for offset, values in enumerate(block):
                yield start + offset, values.astype(np.float64)

    return from_row_scores(len(unit), k, rows())
