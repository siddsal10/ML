"""Download the public real-world datasets used for training/evaluation.

Nothing is redistributed with the repo - this script fetches each dataset
from its official/public source into backend/data/external/ (git-ignored).

    python scripts/download_datasets.py            # all
    python scripts/download_datasets.py hasy mnist # a subset

Datasets
--------
hasy        HASYv2 handwritten symbols (Thoma, 2017). 168,233 32x32 images from
            hundreds of writers, 369 classes incl. digits, + - = ( ) x / \\times.
            Licence: ODbL 1.0. https://zenodo.org/record/259444
mnist       MNIST handwritten digits (LeCun et al., 1998) via OpenML (id 554).
            70,000 28x28 digits written by ~500 writers.
crohme      CROHME handwritten math expressions (Mouchère et al., ICDAR competitions),
            offline-rendered images + LaTeX labels, as mirrored on Hugging Face
            (Neeze/CROHME-full): train 8,834 expressions; test splits 2014/2016/2019.
            Research use only - cite the CROHME competition papers.
assistments ASSISTments 2009-2010 skill-builder logs (Feng, Heffernan & Koedinger,
            2009), as mirrored by the pyBKT examples repository. Real student
            right/wrong answers with skill tags - the standard benchmark for
            knowledge tracing. Free for research use; cite the ASSISTments papers.
"""
from __future__ import annotations

import shutil
import ssl
import sys
import tarfile
import urllib.request
from pathlib import Path

EXTERNAL = Path(__file__).resolve().parents[1] / "data" / "external"

SOURCES = {
    "hasy": ("https://zenodo.org/api/records/259444/files/HASYv2.tar.bz2/content", "HASYv2.tar.bz2"),
    "mnist": ("https://www.openml.org/data/v1/download/52667/mnist_784.arff", "mnist_784.arff"),
    "crohme": [
        (f"https://huggingface.co/datasets/Neeze/CROHME-full/resolve/main/data/{name}.parquet", f"crohme/{name}.parquet")
        for name in ("train-00000-of-00002", "train-00001-of-00002", "2014-00000-of-00001", "2016-00000-of-00001", "2019-00000-of-00001")
    ],
    "assistments": ("https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/as.csv", "assistments_2009.csv"),
}


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already have {dest.name}")
        return
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:  # python.org macOS builds ship without root certificates
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=120, context=ctx) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)
    print(f"  saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def _safe_extract(archive: Path, target: Path) -> None:
    """Extract a tar archive, refusing members that would escape `target`."""
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            dest = (target / m.name).resolve()
            if not str(dest).startswith(str(root)) or m.issym() or m.islnk():
                raise ValueError(f"unsafe path in archive: {m.name}")
        tar.extractall(target)


def main(names: list[str]) -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    for name in names or list(SOURCES):
        print(f"[{name}]")
        entries = SOURCES[name] if isinstance(SOURCES[name], list) else [SOURCES[name]]
        for url, filename in entries:
            dest = EXTERNAL / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            _download(url, dest)
        if name == "hasy" and not (EXTERNAL / "hasy" / "hasy-data-labels.csv").exists():
            print("  extracting ...")
            _safe_extract(dest, EXTERNAL / "hasy")
    print("done ->", EXTERNAL)


if __name__ == "__main__":
    main(sys.argv[1:])
