"""Classical page -> lines -> symbols segmentation (OpenCV, no learning).

1. Page binarization: divide out uneven lighting (phone photos), Otsu
   threshold (ink = 255), then erase ruled notebook lines.
2. Line segmentation from the horizontal projection profile.
3. Symbol segmentation by connected components within each line; components
   that overlap horizontally (the two bars of '=', a broken '5') are merged
   into one symbol.

Training crops are harvested with plain `binarize`; the photo steps in
`binarize_page` are identity on clean scans, so features are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MERGE_OVERLAP = 0.5


@dataclass
class SymbolCrop:
    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray  # ink=255, tight crop containing only this symbol's pixels


def binarize(gray: np.ndarray, normalize: bool = False) -> np.ndarray:
    if normalize:
        gray = flatten_background(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def flatten_background(gray: np.ndarray) -> np.ndarray:
    """Divide the image by an estimate of the paper's brightness, so shadows
    and lighting gradients become uniform white. A grey-level closing much
    wider than any pen stroke removes the ink and leaves the paper. Evenly
    lit images (scans, tablet renders) are returned unchanged."""
    k = max(31, min(gray.shape[:2]) // 6) | 1
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    if np.percentile(background, 99) - np.percentile(background, 1) < 25:
        return gray
    background = cv2.GaussianBlur(background, (k, k), 0).astype(np.float32)
    flat = gray.astype(np.float32) / np.maximum(background, 1.0) * 255.0
    return np.clip(flat, 0, 255).astype(np.uint8)


def remove_ruled_lines(binary: np.ndarray) -> np.ndarray:
    """Erase notebook rules and margins: straight ink runs spanning at least
    half the page and several symbol heights. A minus or '=' bar is never
    that long relative to the symbols around it."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary
    h, w = binary.shape
    sym_h = float(np.median(stats[1:, cv2.CC_STAT_HEIGHT]))
    out = binary.copy()
    for kx, ky in ((max(w // 2, int(4 * sym_h)), 1), (1, max(h // 2, int(4 * sym_h)))):
        if kx > w or ky > h:
            continue
        rules = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
        # thicken slightly so the rule's anti-aliased edges go too
        out[cv2.dilate(rules, np.ones((3, 3), np.uint8)) > 0] = 0
    return out


# Older name kept for callers written against the first photo-robustness study.
def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    return flatten_background(gray)


def binarize_page(gray: np.ndarray, normalize: bool = True) -> np.ndarray:
    return remove_ruled_lines(binarize(gray, normalize=normalize))


def segment_lines(binary: np.ndarray) -> list[tuple[int, int]]:
    row_ink = np.count_nonzero(binary, axis=1)
    min_ink = max(1, int(0.002 * binary.shape[1]))
    rows = row_ink >= min_ink

    runs: list[list[int]] = []
    start = None
    for i, on in enumerate(rows):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(rows)])
    if not runs:
        return []

    # Merge runs separated by a small gap (e.g. a dot or stroke detached
    # from its line); gaps between real lines are much larger.
    median_h = float(np.median([b - a for a, b in runs]))
    merged = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] < 0.25 * median_h:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= max(4, 0.25 * median_h)]


def segment_symbols(line: np.ndarray) -> list[SymbolCrop]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(line, connectivity=8)
    line_h = line.shape[0]
    min_area = max(6, int(0.004 * line_h * line_h))

    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comps.append({"labels": [i], "x0": x, "x1": x + w, "y0": y, "y1": y + h})
    comps.sort(key=lambda c: c["x0"])

    groups: list[dict] = []
    for c in comps:
        if groups:
            g = groups[-1]
            overlap = min(g["x1"], c["x1"]) - max(g["x0"], c["x0"])
            narrower = min(g["x1"] - g["x0"], c["x1"] - c["x0"])
            if narrower > 0 and overlap / narrower >= MERGE_OVERLAP:
                g["labels"] += c["labels"]
                g["x0"], g["x1"] = min(g["x0"], c["x0"]), max(g["x1"], c["x1"])
                g["y0"], g["y1"] = min(g["y0"], c["y0"]), max(g["y1"], c["y1"])
                continue
        groups.append(dict(c))

    crops = []
    for g in groups:
        region = labels[g["y0"] : g["y1"], g["x0"] : g["x1"]]
        mask = np.isin(region, g["labels"]).astype(np.uint8) * 255
        crops.append(SymbolCrop(int(g["x0"]), int(g["y0"]), int(g["x1"] - g["x0"]), int(g["y1"] - g["y0"]), mask))
    return crops


def segment_page(gray: np.ndarray, normalize: bool = True) -> list[list[SymbolCrop]]:
    """Returns the symbol crops of each text line, top to bottom, left to right.
    `normalize=False` skips illumination flattening (ablation only)."""
    binary = binarize_page(gray, normalize=normalize)
    result = []
    for y0, y1 in segment_lines(binary):
        line = binary[y0:y1]
        crops = segment_symbols(line)
        if not crops:
            continue
        for c in crops:
            c.y += y0
        result.append(crops)
    return result


# --- Segmentation hypotheses for recognition-driven decoding ----------------
#
# Connected components are not always symbols: neighbours touch ('12' written
# as one blob) and symbols break ('5' with a detached bar, a two-stroke 'x'
# that does not overlap enough). Instead of committing to one grouping, each
# line yields a lattice: nodes are the gaps between ordered pieces, and an
# edge (i, j) is the symbol made of pieces i..j-1. Pieces are connected
# components, plus the parts of a wide component cut at its thinnest columns.
# The recognizer scores every edge and picks the best path
# (SymbolRecognizer, decode_lattice).

MAX_GROUP = 4      # pieces merged into one symbol at most
MAX_CUTS = 3       # cut points tried inside one component


@dataclass
class Lattice:
    n_nodes: int
    edges: list[tuple[int, int, SymbolCrop]]
    boundary_cost: np.ndarray  # per inner node: 0, or 'overlap' (1) / 'cut' (2) marker
    default_path: list[int]    # node indices of the plain segment_symbols grouping


def _cut_columns(comp_mask: np.ndarray, ref_h: float) -> list[int]:
    """Candidate cut columns inside one wide component: local minima of the
    column ink count, away from the edges and from each other."""
    h, w = comp_mask.shape
    if w < 0.7 * ref_h:
        return []
    col = np.count_nonzero(comp_mask, axis=0).astype(np.float32)
    col = np.convolve(col, np.ones(3, np.float32) / 3, mode="same")
    margin = max(2, int(0.2 * ref_h))
    cand = [c for c in range(margin, w - margin)
            if col[c] <= col[c - 1] and col[c] <= col[c + 1] and col[c] <= 0.5 * col.max()]
    cand.sort(key=lambda c: col[c])
    chosen: list[int] = []
    for c in cand:
        if all(abs(c - o) >= margin for o in chosen):
            chosen.append(c)
        if len(chosen) == MAX_CUTS:
            break
    return sorted(chosen)


def union_mask(parts: list[tuple[int, int, np.ndarray]]) -> tuple[int, int, np.ndarray]:
    """Combine (x, y, mask) parts into one tight (x, y, mask)."""
    x0 = min(x for x, _, _ in parts)
    y0 = min(y for _, y, _ in parts)
    x1 = max(x + m.shape[1] for x, _, m in parts)
    y1 = max(y + m.shape[0] for _, y, m in parts)
    out = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for x, y, m in parts:
        out[y - y0:y - y0 + m.shape[0], x - x0:x - x0 + m.shape[1]] |= m
    return x0, y0, out


def line_lattice(line: np.ndarray) -> Lattice:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(line, connectivity=8)
    line_h = line.shape[0]
    min_area = max(6, int(0.004 * line_h * line_h))
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not keep:
        return Lattice(1, [], np.zeros(1), [0])
    ref_h = float(np.median([stats[i, cv2.CC_STAT_HEIGHT] for i in keep]))
    ref_h = max(ref_h, 0.5 * line_h)

    # pieces: (x, y, mask, component id)
    pieces = []
    for i in keep:
        x, y, w, h = stats[i, :4]
        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        bounds = [0, *_cut_columns(comp, ref_h), w]
        for a, b in zip(bounds, bounds[1:]):
            part = comp[:, a:b]
            rows = np.flatnonzero(part.any(axis=1))
            cols = np.flatnonzero(part.any(axis=0))
            if len(rows) == 0:
                continue
            part = part[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
            pieces.append((int(x + a + cols[0]), int(y + rows[0]), part, i))
    pieces.sort(key=lambda p: p[0])
    m = len(pieces)

    # boundary markers between piece k-1 and k: 1 = they overlap like the bars of '=',
    # 2 = they are parts of one cut component, 0 = a plain gap
    marker = np.zeros(m + 1)
    reach = pieces[0][0] + pieces[0][2].shape[1]
    for k in range(1, m):
        x, _, mk, comp = pieces[k]
        prev_x, _, prev_m, prev_comp = pieces[k - 1]
        if comp == prev_comp:
            marker[k] = 2
        else:
            overlap = reach - x
            narrower = min(mk.shape[1], prev_m.shape[1])
            if narrower > 0 and overlap / narrower >= MERGE_OVERLAP:
                marker[k] = 1
        reach = max(reach, x + mk.shape[1])

    edges = []
    gap_tol = 0.15 * ref_h
    for i in range(m):
        reach = pieces[i][0] + pieces[i][2].shape[1]
        for j in range(i + 1, min(m, i + MAX_GROUP) + 1):
            if j > i + 1:
                x, _, mk, _ = pieces[j - 1]
                if x > reach + gap_tol:
                    break
                reach = max(reach, x + mk.shape[1])
                if reach - pieces[i][0] > 2.0 * ref_h:
                    break
            gx, gy, gm = union_mask([(p[0], p[1], p[2]) for p in pieces[i:j]])
            edges.append((i, j, SymbolCrop(gx, gy, gm.shape[1], gm.shape[0], gm)))
    # the plain grouping: break wherever segment_symbols would (no cuts, overlap merges)
    default_path = [0] + [k for k in range(1, m) if marker[k] == 0] + [m]
    return Lattice(m + 1, edges, marker, default_path)


def page_lattices(gray: np.ndarray, normalize: bool = True) -> list[tuple[int, Lattice]]:
    """(line top y, lattice) per text line, top to bottom."""
    binary = binarize_page(gray, normalize=normalize)
    out = []
    for y0, y1 in segment_lines(binary):
        lat = line_lattice(binary[y0:y1])
        if lat.edges:
            for _, _, c in lat.edges:
                c.y += y0
            out.append((y0, lat))
    return out
