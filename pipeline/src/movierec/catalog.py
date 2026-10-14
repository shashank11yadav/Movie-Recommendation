"""The movie catalog and the single mapping between MovieLens ids and matrix rows.

Every model in this package works on dense row indices. The old version of this
project broke because it mixed DataFrame labels with positions after filtering
rows. Here the mapping is built once, from a sorted id array, and everything
else goes through ``Catalog.row_of`` / ``Catalog.ids``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

NO_GENRE = "(no genres listed)"
MAX_TAGS_PER_MOVIE = 8
MAX_TAG_LENGTH = 40
# Tags about someone's viewing logistics rather than the film, plus profanity
# that would otherwise show up in explanation chips.
STOP_TAGS = frozenset(
    {
        "in netflix queue",
        "not available from netflix",
        "no dvd at netflix",
        "imdb top 250",
        "seen at the cinema",
        "seen more than once",
        "not seen",
        "mindfuck",
        "fucked up",
        "motherfucker",
    }
)

# Articles MovieLens moves to the end of a title ("Matrix, The").
_ARTICLES = (
    "The", "A", "An", "Les", "La", "Le", "L'", "Il", "Das", "Der", "Die",
    "El", "Los", "Las", "Un", "Une", "Lo", "I", "Gli", "Det", "Den", "O",
)  # fmt: skip
_YEAR_RE = re.compile(r"\s*\((\d{4})(?:[-\u2013]\d{0,4})?\)\s*$")
_ARTICLE_RE = re.compile(
    r"^(?P<body>.+), (?P<article>" + "|".join(map(re.escape, _ARTICLES)) + ")$"
)


def parse_title(raw: str) -> tuple[str, int | None]:
    """Split "Matrix, The (1999)" into ("The Matrix", 1999).

    Alternate titles in parentheses are kept, and only the main title gets its
    article moved back to the front.
    """
    text = raw.strip()
    year: int | None = None
    if match := _YEAR_RE.search(text):
        year = int(match.group(1))
        text = text[: match.start()].strip()

    main, sep, rest = text.partition(" (")
    if article_match := _ARTICLE_RE.match(main):
        article = article_match.group("article")
        joiner = "" if article.endswith("'") else " "
        main = f"{article}{joiner}{article_match.group('body')}"
    return (main + sep + rest if sep else main), year


def clean_tag(tag: str) -> str | None:
    """Normalise a free-text user tag, or drop it if nothing useful is left."""
    text = unicodedata.normalize("NFKC", tag).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if not text or len(text) > MAX_TAG_LENGTH or not any(ch.isalnum() for ch in text):
        return None
    if text in STOP_TAGS:
        return None
    return text


def top_tags(tags: Iterable[str], limit: int = MAX_TAGS_PER_MOVIE) -> list[str]:
    cleaned = [t for t in (clean_tag(tag) for tag in tags) if t is not None]
    counts = Counter(cleaned)
    # Most-applied first, then alphabetical so the order is stable between runs.
    return [tag for tag, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]


@dataclass(frozen=True)
class Catalog:
    ids: npt.NDArray[np.int64]
    titles: list[str]
    years: list[int | None]
    genres: list[list[str]]
    tags: list[list[str]]
    _row: dict[int, int] = field(repr=False)

    @classmethod
    def build(cls, movies: pd.DataFrame, tags: pd.DataFrame) -> Catalog:
        frame = movies.sort_values("movieId", kind="stable").reset_index(drop=True)
        ids = frame["movieId"].to_numpy(dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("movie ids must be unique")

        parsed = [parse_title(str(t)) for t in frame["title"]]
        genres = [[] if str(g) == NO_GENRE else sorted(str(g).split("|")) for g in frame["genres"]]
        raw_tags: dict[int, list[str]] = defaultdict(list)
        for mid, tag in zip(
            tags["movieId"].to_numpy(dtype=np.int64).tolist(),
            tags["tag"].astype(str).tolist(),
            strict=True,
        ):
            raw_tags[mid].append(tag)
        tags_by_movie = {mid: top_tags(values) for mid, values in raw_tags.items()}
        return cls(
            ids=ids,
            titles=[title for title, _ in parsed],
            years=[year for _, year in parsed],
            genres=genres,
            tags=[tags_by_movie.get(int(mid), []) for mid in ids],
            _row={int(mid): row for row, mid in enumerate(ids)},
        )

    def __len__(self) -> int:
        return len(self.ids)

    def row_of(self, movie_id: int) -> int:
        try:
            return self._row[movie_id]
        except KeyError:
            raise KeyError(f"unknown movieId {movie_id}") from None

    def rows_of(self, movie_ids: Sequence[int] | npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        return np.fromiter((self.row_of(int(m)) for m in movie_ids), dtype=np.int64)

    def label(self, row: int) -> str:
        year = self.years[row]
        return f"{self.titles[row]} ({year})" if year else self.titles[row]

    def find(self, query: str) -> list[int]:
        """Case- and accent-insensitive substring search, most specific match first."""
        needle = _fold(query)
        if not needle:
            return []
        hits = [row for row in range(len(self)) if needle in _fold(self.label(row))]
        return sorted(hits, key=lambda r: (len(self.label(r)), r))


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold().strip()
