"""Whole-line recognizer: fully convolutional network + CTC (no symbol segmentation).

Segmenting a line into symbols before reading them fails on real photos, where
students' digits touch (HME100K test photos: the segmenter finds the right number
of symbols in only 18% of lines). This model reads the whole line image and emits
a token sequence, trained with Connectionist Temporal Classification, so it learns
from every linear expression in the training data, not only those whose symbols
could be count-aligned.

Input: the photo-robust binary page (segmentation.binarize_page), one text line,
cropped to its ink and scaled to height 32 (ink = 1). Output: per column of 4
pixels, a distribution over the token vocabulary + CTC blank. The vocabulary holds
every token seen in training (letters, '.', ...), not only StepWise's 17 classes:
more data to learn from, and an out-of-vocabulary read is reported as such.

The network has no recurrent layers (dilated 1-D convolutions give context), so it
exports to ONNX and runs in OpenCV's DNN module like the symbol CNN.
"""
from __future__ import annotations

import io
import math
import time

import cv2
import numpy as np

HEIGHT = 32
MAX_WIDTH = 640
STRIDE = 4
BLANK = 0


def line_image(binary_line: np.ndarray) -> np.ndarray | None:
    """Binary line (ink=255) -> float32 (HEIGHT, W) in [0, 1], cropped to the ink, W a multiple of STRIDE."""
    r = line_image_with_box(binary_line)
    return None if r is None else r[0]


def frame_columns(f0: int, f1: int, col0: int, scale: float) -> tuple[int, int]:
    """Frames [f0, f1) of a line image -> column range in the binary line it came from."""
    return int(col0 + max(0.0, (f0 * STRIDE - 4) / scale)), int(math.ceil(col0 + (f1 * STRIDE - 4) / scale))


def line_image_with_box(binary_line: np.ndarray):
    """(image, first ink column, scale) so frames can be mapped back to the line; None if empty."""
    rows = np.flatnonzero(binary_line.any(axis=1))
    cols = np.flatnonzero(binary_line.any(axis=0))
    if len(rows) == 0:
        return None
    crop = binary_line[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = crop.shape
    scale = (HEIGHT - 6) / h
    new_w = int(min(MAX_WIDTH - 8, max(4, round(w * scale))))
    img = cv2.resize(crop, (new_w, HEIGHT - 6), interpolation=cv2.INTER_AREA)
    width = int(math.ceil((new_w + 8) / STRIDE) * STRIDE)
    out = np.zeros((HEIGHT, width), np.float32)
    out[3:3 + img.shape[0], 4:4 + new_w] = img.astype(np.float32) / 255.0
    return out, int(cols[0]), new_w / w


def _build_net(n_out: int):
    import torch.nn as nn

    def conv(cin, cout, k=3, pool=None):
        layers = [nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(pool))
        return layers

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                *conv(1, 32, pool=(2, 2)),                       # 16 x W/2
                *conv(32, 64), *conv(64, 64, pool=(2, 2)),       # 8 x W/4
                *conv(64, 128), *conv(128, 128, pool=(2, 1)),    # 4 x W/4
                *conv(128, 192), nn.MaxPool2d((4, 1)),           # 1 x W/4
            )
            ctx = []
            for d in (1, 2, 4, 8, 1):
                ctx += [nn.Conv2d(192, 192, (1, 3), padding=(0, d), dilation=(1, d), bias=False),
                        nn.BatchNorm2d(192), nn.ReLU(inplace=True)]
            self.ctx = nn.Sequential(*ctx)
            self.drop = nn.Dropout(0.2)
            self.head = nn.Conv2d(192, n_out, 1)

        def forward(self, x):  # (B,1,32,W) -> (B,n_out,1,W/4)
            f = self.cnn(x)
            f = f + self.ctx(f)
            return self.head(self.drop(f))

    return Net()


def _augment(x, gen):
    """Mild random affine warp + pen width on a (B,1,H,W) batch."""
    import torch
    import torch.nn.functional as F

    b = x.shape[0]
    u = lambda lo, hi: torch.empty(b).uniform_(lo, hi, generator=gen)  # noqa: E731
    shear, sy, ang = u(-0.2, 0.2), u(0.9, 1.1), u(-0.03, 0.03)
    theta = torch.zeros(b, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1] = 1.0, shear * x.shape[2] / x.shape[3]
    theta[:, 1, 1], theta[:, 1, 0] = sy, ang * x.shape[3] / x.shape[2]
    theta[:, 1, 2] = u(-0.06, 0.06)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    pen = torch.rand(b, generator=gen).view(-1, 1, 1, 1)
    x = torch.where(pen < 0.2, F.max_pool2d(x, 3, 1, 1), x)
    return torch.where(pen > 0.9, -F.max_pool2d(-x, (3, 1), 1, (1, 0)), x)


class LineCTCRecognizer:
    def __init__(self, epochs: int = 20, batch_size: int = 32, lr: float = 2e-3, seed: int = 0):
        self.epochs, self.batch_size, self.lr, self.seed = epochs, batch_size, lr, seed

    def fit(self, images: list[np.ndarray], labels: list[list[str]], val=None, log=print, max_hours: float | None = None,
            init: "LineCTCRecognizer | None" = None):
        """images: line images from line_image(); labels: token lists. val: (images, labels) for progress.
        init: a trained recognizer to continue from (its vocabulary is kept, new tokens are appended)."""
        import torch

        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        vocab = sorted({t for lab in labels for t in lab})
        if init is not None:
            vocab = init.tokens_[1:] + [t for t in vocab if t not in set(init.tokens_)]
        self.tokens_ = ["<blank>"] + vocab
        index = {t: i for i, t in enumerate(self.tokens_)}
        keep = [i for i, (im, lab) in enumerate(zip(images, labels)) if 0 < len(lab) <= im.shape[1] // STRIDE // 2]
        order_by_w = sorted(keep, key=lambda i: images[i].shape[1])
        batches = [order_by_w[s:s + self.batch_size] for s in range(0, len(order_by_w), self.batch_size)]
        net = _build_net(len(self.tokens_))
        if init is not None:
            old = torch.load(io.BytesIO(init.state_), weights_only=True)
            n = len(init.tokens_)
            new = net.state_dict()
            for k, v in old.items():
                if k.startswith("head."):
                    new[k][:n] = v
                else:
                    new[k] = v
            net.load_state_dict(new)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=len(batches) * self.epochs, pct_start=0.15)
        ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
        t0 = time.perf_counter()
        for ep in range(self.epochs):
            net.train()
            total = 0.0
            for bi in torch.randperm(len(batches), generator=gen).tolist():
                ids = batches[bi]
                w = max(images[i].shape[1] for i in ids)
                x = np.zeros((len(ids), 1, HEIGHT, w), np.float32)
                for k, i in enumerate(ids):
                    im = images[i]
                    x[k, 0, :, :im.shape[1]] = im / 255.0 if im.dtype == np.uint8 else im
                xb = _augment(torch.from_numpy(x), gen)
                logp = net(xb).squeeze(2).permute(2, 0, 1).log_softmax(2)  # (T,B,C)
                tgt = torch.tensor([index[t] for i in ids for t in labels[i]])
                tl = torch.tensor([len(labels[i]) for i in ids])
                il = torch.tensor([images[i].shape[1] // STRIDE for i in ids])
                loss = ctc(logp, tgt, il, tl)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                sched.step()
                total += float(loss.detach())
            self._export(net)
            msg = f"epoch {ep + 1}/{self.epochs} loss {total / len(batches):.4f} ({time.perf_counter() - t0:.0f}s)"
            if val is not None:
                hyp = [self.decode(im)[0] for im in val[0]]
                msg += f" val exact {np.mean([h == list(r) for h, r in zip(hyp, val[1])]):.4f}"
            log(msg)
            if max_hours and time.perf_counter() - t0 > max_hours * 3600:
                log("  time budget reached")
                break
        return self

    def _export(self, net) -> None:
        import torch

        net.eval()
        state = io.BytesIO()
        torch.save(net.state_dict(), state)
        self.state_ = state.getvalue()
        buf = io.BytesIO()
        torch.onnx.export(net, torch.zeros(1, 1, HEIGHT, 128), buf, input_names=["line"], output_names=["logits"],
                          dynamic_axes={"line": {3: "width"}, "logits": {3: "frames"}}, opset_version=13, dynamo=False)
        self.onnx_ = buf.getvalue()
        self._net = None

    def _dnn(self):
        if getattr(self, "_net", None) is None:
            self._net = cv2.dnn.readNetFromONNX(np.frombuffer(self.onnx_, np.uint8))
        return self._net

    def frame_probs(self, img: np.ndarray) -> np.ndarray:
        """(frames, tokens) softmax probabilities for one line image."""
        net = self._dnn()
        net.setInput(img[None, None].astype(np.float32))
        z = net.forward()[0, :, 0, :].T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def decode(self, img: np.ndarray) -> tuple[list[str], list[dict]]:
        """Greedy CTC decoding: tokens, and per token its frame span and confidence."""
        p = self.frame_probs(img)
        best = p.argmax(axis=1)
        tokens, spans, prev = [], [], BLANK
        for f, c in enumerate(best):
            if c != BLANK and c != prev:
                tokens.append(self.tokens_[c])
                spans.append({"f0": f, "f1": f + 1, "conf": float(p[f, c])})
            elif c != BLANK and c == prev:
                spans[-1]["f1"] = f + 1
                spans[-1]["conf"] = max(spans[-1]["conf"], float(p[f, c]))
            prev = c
        return tokens, spans

    def decode_constrained(self, img: np.ndarray, allowed: dict[str, str], step, final, init,
                           beam: int = 16, top_k: int = 5) -> tuple[list[str], list[dict]] | None:
        """Best CTC path whose tokens are all in `allowed` (token -> output symbol) and whose
        symbol sequence the grammar accepts: `step(state, symbol)` gives the next state or None,
        `final(state)` says whether a line may end there. Beam search over best paths, so each
        token keeps its frame span. Returns (symbols, spans), or None if nothing is accepted."""
        p = self.frame_probs(img)
        ok = [i for i, t in enumerate(self.tokens_) if i != BLANK and t in allowed]
        q = p[:, [BLANK] + ok]
        logq = np.log(q / q.sum(axis=1, keepdims=True) + 1e-12)
        idx = [BLANK] + ok
        # key (symbols, last frame label) -> (score, grammar state, spans)
        beams: dict[tuple, tuple[float, object, list]] = {((), BLANK): (0.0, init, [])}
        for f in range(len(logq)):
            cand = np.argsort(logq[f])[::-1][:top_k]
            if 0 not in cand:
                cand = np.append(cand, 0)
            nxt: dict[tuple, tuple[float, object, list]] = {}

            def push(key, score, state, spans):
                if key not in nxt or nxt[key][0] < score:
                    nxt[key] = (score, state, spans)

            for (syms, last), (score, state, spans) in beams.items():
                for j in cand:
                    s = score + float(logq[f, j])
                    lab = idx[j]
                    if lab == BLANK:
                        push((syms, BLANK), s, state, spans)
                    elif lab == last:  # same token continues over this frame
                        sp = spans[-1]
                        push((syms, lab), s, state, spans[:-1] + [{**sp, "f1": f + 1,
                                                                   "conf": max(sp["conf"], float(p[f, lab]))}])
                    else:
                        sym = allowed[self.tokens_[lab]]
                        st = step(state, sym)
                        if st is not None:
                            push((syms + (sym,), lab), s, st, spans + [{"f0": f, "f1": f + 1, "conf": float(p[f, lab])}])
            beams = dict(sorted(nxt.items(), key=lambda kv: -kv[1][0])[:beam])
        done = [(v[0], k[0], v[2]) for k, v in beams.items() if k[0] and final(v[1])]
        if not done:
            return None
        _, syms, spans = max(done, key=lambda d: d[0])
        return list(syms), spans

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_net"}
