#!/usr/bin/env python
"""Download the training corpus and record its provenance.

Run once, before ``rc-build-dataset``:

    python scripts/download_corpus.py

The corpus is not committed to the repository: it is 56 MB, it is third-party
data under its own licence, and it contains real people's resumes. What *is*
committed is the manifest this script writes, which records the exact URL,
retrieval time, byte count and SHA-256 of every file -- so the build stays
reproducible without the bytes living in git.

No credentials and no API key are required.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE = "livecareer_resumes"
SOURCE_VERSION = "v1"
HUB_REPO = "https://huggingface.co/datasets/Darshan-04/Resume-classification"
BASE_URL = f"{HUB_REPO}/resolve/main"
FILES = ("Resume.csv",)

#: Verified at download time. A mismatch means the upstream file changed, which
#: would silently change every downstream metric.
EXPECTED_SHA256 = {
    "Resume.csv": "816a7cd985a9a41e3ce8d1e83cb7e1beb1e854a89c1cf1fd0d75494faaa0559d",
}


def _download(url: str, destination: Path) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "resume-classifier/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return len(payload), hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="project data directory (default: ./data)",
    )
    args = parser.parse_args(argv)

    destination = args.data_dir / "raw" / SOURCE / SOURCE_VERSION
    destination.mkdir(parents=True, exist_ok=True)

    print(f"source : {HUB_REPO}")
    print(f"target : {destination}")
    print()

    records = []
    for name in FILES:
        path = destination / name
        if path.is_file() and not args.force:
            size = path.stat().st_size
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f"  {name}: already present ({size:,} bytes), skipping")
        else:
            print(f"  {name}: downloading...", flush=True)
            try:
                size, digest = _download(f"{BASE_URL}/{name}", path)
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"\nerror: download failed: {exc}", file=sys.stderr)
                print(
                    "The corpus can also be fetched manually from:\n"
                    f"  {HUB_REPO}\n"
                    f"Place Resume.csv in {destination}",
                    file=sys.stderr,
                )
                return 2
            print(f"  {name}: {size:,} bytes")

        expected = EXPECTED_SHA256.get(name)
        if expected and digest != expected:
            print(
                f"\nerror: checksum mismatch for {name}\n"
                f"  expected {expected}\n  got      {digest}\n"
                "The upstream file has changed. Every downstream metric would "
                "shift silently, so the download is being refused.",
                file=sys.stderr,
            )
            return 3

        records.append({"file": name, "bytes": size, "sha256": digest, "url": f"{BASE_URL}/{name}"})

    manifest = {
        "source": SOURCE,
        "source_version": SOURCE_VERSION,
        "retrieved_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "retrieval_method": "direct HTTPS download, no authentication, no API key",
        "hub_repo": HUB_REPO,
        "declared_license": "mit",
        "upstream_origin": (
            "Mirror of the Kaggle dataset 'snehaanbhawal/resume-dataset', scraped "
            "from publicly published resume examples on livecareer.com. Contains "
            "real resume text: treat as personal data."
        ),
        "files": records,
    }
    (destination / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print()
    print("checksums verified; MANIFEST.json written")
    print("next: rc-build-dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
