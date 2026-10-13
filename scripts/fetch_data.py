"""Fetch the raw inputs, or copy a dated capture, then verify it is *the* capture.

Nothing under ``data/raw/`` is committed: the sources carry different and partly
restrictive redistribution terms, so this repository ships the recipe, not the
bytes. The URLs and file names are those of phishing-url-detector, whose dataset
code is vendored unchanged.

The phishing feeds are live and rotate within hours. The published comparison is
only reproducible against the 2026-08-14 capture, so every file is checked
against ``data/SHA256SUMS``. A fresh download today will differ for the feeds;
the script says so and exits non-zero rather than letting different data pass as
the same experiment.

Usage::

    uv run python scripts/fetch_data.py                       # download
    uv run python scripts/fetch_data.py --copy-from ../phishing-url-detector/data/raw
    uv run python scripts/fetch_data.py --accept-different-capture
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = RAW / "external"
SNAPSHOT = "2026-08-14"
PHIUSIIL_CSV = "PhiUSIIL_Phishing_URL_Dataset.csv"

SOURCES: list[tuple[str, str, str]] = [
    (
        "PhiUSIIL_Phishing_URL_Dataset.csv.zip",
        "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip",
        "PhiUSIIL (UCI id 967, CC BY 4.0) — the training source",
    ),
    (
        f"external/openphish-{SNAPSHOT}.txt",
        "https://openphish.com/feed.txt",
        "OpenPhish community feed — independent phishing probe",
    ),
    (
        f"external/phishtank-{SNAPSHOT}.csv.gz",
        "https://data.phishtank.com/data/online-valid.csv.gz",
        "PhishTank verified dump — submission times enable the temporal test",
    ),
    (
        f"external/tranco-{SNAPSHOT}.csv.zip",
        "https://tranco-list.eu/top-1m.csv.zip",
        "Tranco top 1M — benign side of the OOD test",
    ),
]


def read_manifest() -> dict[str, str]:
    manifest = {}
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            manifest[name.strip()] = digest
    return manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, url: str, description: str, force: bool) -> None:
    destination = RAW / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"  present  {name}")
        return
    print(f"  fetching {name}  — {description}")
    try:
        with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 - fixed HTTPS URLs
            destination.write_bytes(response.read())
    except OSError as error:
        raise SystemExit(f"  FAILED   {name}: {type(error).__name__}: {error}") from error


def copy_capture(source: Path) -> None:
    """Copy an existing dated capture, e.g. a phishing-url-detector checkout's data/raw."""
    if not source.is_dir():
        raise SystemExit(f"--copy-from: {source} is not a directory")
    wanted = [PHIUSIIL_CSV, "PhiUSIIL_Phishing_URL_Dataset.csv.zip"] + [name for name, _, _ in SOURCES[1:]]
    for name in wanted:
        origin = source / name
        if origin.exists():
            (RAW / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, RAW / name)
            print(f"  copied   {name}")


def extract_phiusiil() -> None:
    csv_path = RAW / PHIUSIIL_CSV
    zip_path = RAW / "PhiUSIIL_Phishing_URL_Dataset.csv.zip"
    if csv_path.exists() or not zip_path.exists():
        return
    with zipfile.ZipFile(zip_path) as archive:
        member = next(n for n in archive.namelist() if n.endswith(".csv"))
        csv_path.write_bytes(archive.read(member))


def verify() -> list[str]:
    """Names whose content differs from the 2026-08-14 capture (missing files included)."""
    differing = []
    for name, expected in read_manifest().items():
        path = RAW / name
        if not path.exists():
            print(f"  MISSING  {name}")
            differing.append(name)
        elif sha256(path) != expected:
            print(f"  DIFFERS  {name}")
            differing.append(name)
        else:
            print(f"  ok       {name}")
    return differing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--copy-from", type=Path, help="copy a dated capture instead of downloading")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--accept-different-capture", action="store_true",
                        help="continue even if the files are not the 2026-08-14 capture")
    args = parser.parse_args()

    if args.copy_from:
        copy_capture(args.copy_from)
    else:
        for name, url, description in SOURCES:
            download(name, url, description, args.force)
    extract_phiusiil()

    print(f"\nVerifying against the {SNAPSHOT} capture (data/SHA256SUMS):")
    differing = verify()
    if differing and not args.accept_different_capture:
        print(
            f"\n{len(differing)} file(s) are not the {SNAPSHOT} capture. The phishing feeds rotate, so a "
            "fresh download cannot reproduce the published numbers exactly, and the baseline check in "
            "`make eval` will refuse to run. Copy the original capture with --copy-from, or pass "
            "--accept-different-capture to study a new snapshot knowingly.",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
