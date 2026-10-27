from __future__ import annotations

import hashlib
import io
import ssl
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from movierec import config, data

CSV = {
    "movies.csv": (
        "movieId,title,genres\n1,Toy Story (1995),Animation|Children\n2,Heat (1995),Crime\n"
    ),
    "ratings.csv": "userId,movieId,rating,timestamp\n1,1,4.0,10\n1,2,3.5,11\n",
    "tags.csv": "userId,movieId,tag,timestamp\n1,1,pixar,12\n",
    "links.csv": "movieId,imdbId,tmdbId\n1,114709,862\n",
    "README.txt": "test",
}


def make_zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


def archive_files(**overrides: str) -> dict[str, str]:
    files = {f"{config.DATASET_ARCHIVE_DIR}/{k}": v for k, v in CSV.items()}
    files.update(overrides)
    return files


def fake_fetch(payload: bytes) -> Callable[[str, Path, int], None]:
    def _fetch(url: str, dest: Path, max_bytes: int) -> None:
        dest.write_bytes(payload)

    return _fetch


def test_download_verifies_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = make_zip(archive_files())
    monkeypatch.setattr(data, "_fetch", fake_fetch(payload))
    good = hashlib.sha256(payload).hexdigest()
    path = data.download(tmp_path, expected_sha256=good)
    assert path.read_bytes() == payload

    with pytest.raises(data.DatasetError, match="checksum mismatch"):
        data.download(tmp_path / "other", expected_sha256="0" * 64)
    assert not list((tmp_path / "other").glob("*.part"))


def test_fetch_refuses_plain_http(tmp_path: Path) -> None:
    with pytest.raises(data.DatasetError, match="non-HTTPS"):
        data._fetch("http://example.com/x.zip", tmp_path / "x", 10)


def test_grouplens_ca_matches_its_pinned_fingerprint() -> None:
    der = ssl.PEM_cert_to_DER_cert(config.GROUPLENS_CA_PEM)
    assert hashlib.sha256(der).hexdigest() == config.GROUPLENS_CA_SHA256
    context = data._ssl_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


def test_extract_only_writes_expected_members(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    archive.write_bytes(make_zip(archive_files(**{"../../evil.sh": "rm -rf /"})))
    out = data.extract(archive, tmp_path / "out")
    assert sorted(p.name for p in out.iterdir()) == sorted(CSV)
    assert not (tmp_path / "evil.sh").exists()


def test_extract_fails_on_missing_member(tmp_path: Path) -> None:
    files = archive_files()
    del files[f"{config.DATASET_ARCHIVE_DIR}/ratings.csv"]
    archive = tmp_path / "a.zip"
    archive.write_bytes(make_zip(files))
    with pytest.raises(data.DatasetError, match="missing"):
        data.extract(archive, tmp_path / "out")


def _write(tmp_path: Path, **overrides: str) -> Path:
    for name, text in {**CSV, **overrides}.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_load_reads_valid_files(tmp_path: Path) -> None:
    ds = data.load(_write(tmp_path))
    assert len(ds.movies) == 2
    assert ds.ratings["rating"].tolist() == [4.0, 3.5]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"movies.csv": CSV["movies.csv"] + "2,Heat again (1995),Crime\n"}, "duplicate movieId"),
        ({"ratings.csv": CSV["ratings.csv"] + "1,99,4.0,13\n"}, "missing from movies"),
        ({"ratings.csv": CSV["ratings.csv"] + "1,1,2.0,14\n"}, "repeated"),
        ({"ratings.csv": CSV["ratings.csv"] + "2,1,9.0,14\n"}, "scale"),
    ],
)
def test_load_rejects_bad_data(tmp_path: Path, override: dict[str, str], message: str) -> None:
    with pytest.raises(data.DatasetError, match=message):
        data.load(_write(tmp_path, **override))
