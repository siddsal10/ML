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
scratchmath ScratchMath (songdj/ScratchMath on Hugging Face): 1,700+ REAL student
            handwritten scratchwork photos (primary + middle school) with question,
            correct answer, student's wrong answer and an expert-annotated error
            category (7 classes). Licence: CC BY-NC-SA 4.0 (non-commercial research).
assistments ASSISTments 2009-2010 skill-builder logs (Feng, Heffernan & Koedinger,
            2009), as mirrored by the pyBKT examples repository. Real student
            right/wrong answers with skill tags - the standard benchmark for
            knowledge tracing. Free for research use; cite the ASSISTments papers.
cogtutor    CogTutorBugs (McNichols, Zhang & Lan, 2023): 5,339 real erroneous steps by
            95 Algebra I students in the Cognitive Tutor (DataShop 'Cog Tutor 2010
            Equation Solving'), each an equation, the student's next equation and one
            of 19 error labels. Licence: CC BY 4.0. https://github.com/umass-ml4ed/CogTutorBugs
hme100k     HME100K (Yuan et al., CVPR 2022, TAL Education): 99,109 PHOTOGRAPHS of real
            handwritten math on paper by tens of thousands of students, uploaded from a
            phone app (blur, shadows, ruled paper, varied writers), with LaTeX labels.
            train 74,502 / test 24,607. Unofficial Hugging Face mirror (ConstantHao/HME100K)
            of https://ai.100tal.com/dataset; research use only - cite the paper.
mathwriting MathWriting (Gervais et al., Google, KDD 2025): 230k human-written expressions
            (online inks + LaTeX) from thousands of writers, writer-disjoint train /
            valid / test splits. Licence: CC BY-NC-SA 4.0. Kept as the .tgz and
            streamed (app/ml/real_cnn.py renders the strokes), not extracted.
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
    "scratchmath": [
        (f"https://huggingface.co/datasets/songdj/ScratchMath/resolve/main/{cfg}/data-00000-of-00001.parquet", f"scratchmath/{cfg}.parquet")
        for cfg in ("primary", "middle")
    ],
    "hme100k": [
        (f"https://huggingface.co/datasets/ConstantHao/HME100K/resolve/main/data/{name}.parquet", f"hme100k/{name}.parquet")
        for name in ("test-00000-of-00001", "train-00000-of-00002", "train-00001-of-00002")
    ],
    "mathwriting": ("https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz", "mathwriting-2024.tgz"),
    "assistments": ("https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/as.csv", "assistments_2009.csv"),
    "cogtutor": ("https://raw.githubusercontent.com/umass-ml4ed/CogTutorBugs/main/data/bug_full_5339.json",
                 "cogtutor/bug_full_5339.json"),
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
    with urllib.request.urlopen(url, timeout=600, context=ctx) as r, open(tmp, "wb") as f:
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
