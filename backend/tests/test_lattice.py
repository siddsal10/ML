"""Recognition-driven segmentation: lattice construction and joint decoding."""
import numpy as np
import pytest

from app.ml.segmentation import Lattice, SymbolCrop, line_lattice, union_mask
from app.ml.symbol_classifier import DEFAULT_DECODER, JUNK, decode_lattice, decode_lattice_argmax

CLASSES = sorted(list("0123456789") + ["+", "-", "=", "x", "(", ")", "/", JUNK])


def _crop():
    return SymbolCrop(0, 0, 1, 1, np.full((1, 1), 255, np.uint8))


def _probs(rows):
    """rows: list of {class: prob}; remaining mass spread over the other classes."""
    out = np.full((len(rows), len(CLASSES)), 1e-4)
    for r, d in enumerate(rows):
        for c, p in d.items():
            out[r, CLASSES.index(c)] = p
    return out / out.sum(axis=1, keepdims=True)


def test_touching_blob_gets_cut_hypotheses():
    line = np.zeros((40, 80), np.uint8)
    line[5:35, 5:9] = 255     # '1'
    line[5:35, 25:29] = 255   # another '1' ...
    line[18:20, 9:25] = 255   # ... joined to the first by a thin stroke
    line[5:35, 60:64] = 255   # separate '1'
    lat = line_lattice(line)
    assert lat.n_nodes - 1 >= 3                    # the blob was cut into pieces
    assert any(c == 2 for c in lat.boundary_cost)  # and the cut is marked
    assert lat.default_path == [0, *[k for k in range(1, lat.n_nodes - 1) if lat.boundary_cost[k] == 0], lat.n_nodes - 1]


def test_decoder_prefers_split_when_classifier_is_confident():
    # pieces A, B: the merged crop looks like junk, each half a clear '1'; then '=' '2'
    lat = Lattice(5, [(0, 1, _crop()), (1, 2, _crop()), (0, 2, _crop()), (2, 3, _crop()), (3, 4, _crop())],
                  np.array([0, 2, 0, 0, 0.0]), [0, 2, 3, 4])
    probs = _probs([{"1": 0.95}, {"1": 0.95}, {JUNK: 0.9, "4": 0.05}, {"=": 0.99}, {"2": 0.99}])
    path = decode_lattice(lat, probs, CLASSES, DEFAULT_DECODER)
    assert "".join(CLASSES[c] for _, c in path) == "11=2"
    # the plain grouping keeps the blob whole
    assert "".join(CLASSES[c] for _, c in decode_lattice_argmax(lat, probs, CLASSES)) == "4=2"


def test_decoder_keeps_overlapping_bars_together():
    # two overlapping bars: '=' as one symbol beats '-' '-' (also ill-formed)
    lat = Lattice(5, [(0, 1, _crop()), (1, 2, _crop()), (2, 3, _crop()), (1, 3, _crop()), (3, 4, _crop())],
                  np.array([0, 0, 1, 0, 0.0]), [0, 1, 3, 4])
    probs = _probs([{"3": 0.99}, {"-": 0.9}, {"-": 0.9}, {"=": 0.9}, {"3": 0.99}])
    path = decode_lattice(lat, probs, CLASSES, DEFAULT_DECODER)
    assert "".join(CLASSES[c] for _, c in path) == "3=3"


def test_union_mask_places_parts():
    x, y, m = union_mask([(2, 3, np.full((2, 2), 255, np.uint8)), (6, 1, np.full((1, 1), 255, np.uint8))])
    assert (x, y, m.shape) == (2, 1, (4, 5)) and m.sum() == 5 * 255


def test_cnn_roundtrip_through_onnx():
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    import pickle

    from app.ml.glyph_data import generate_dataset
    from app.ml.symbol_cnn import CNNSymbolClassifier

    d = generate_dataset(samples_per_font=6, seed=0)
    m = CNNSymbolClassifier(epochs=1, batch_size=64).fit(d.X, d.y)
    m2 = pickle.loads(pickle.dumps(m))
    p = m2.predict_proba(d.X[:5])
    assert p.shape == (5, len(m.classes_)) and np.allclose(p.sum(axis=1), 1, atol=1e-5)


def test_forced_alignment_follows_the_known_transcription():
    from app.ml.real_cnn import forced_align

    # blob = pieces 0+1; the model prefers reading it whole as '4', but the label says '11'
    lat = Lattice(4, [(0, 1, _crop()), (1, 2, _crop()), (0, 2, _crop()), (2, 3, _crop())],
                  np.array([0, 2, 0, 0.0]), [0, 2, 3])
    probs = _probs([{"1": 0.7}, {"1": 0.8}, {"4": 0.9}, {"7": 0.95}])
    path = forced_align([(lat, probs)], CLASSES, ["1", "1", "7"])
    assert path is not None and len(path) == 3
    # an alignment that needs implausible reads is rejected
    assert forced_align([(lat, probs)], CLASSES, ["2", "2", "2"]) is None
    # a transcription with more symbols than the lattice can hold is impossible
    assert forced_align([(lat, probs)], CLASSES, ["1", "1", "7", "7"]) is None
