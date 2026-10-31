"""Small convolutional network for single handwritten symbols.

A scikit-learn-style estimator (fit / predict_proba / predict / classes_)
over FEATURE_VERSION 2 vectors, reading only their 32x32 glyph block, so it
is a drop-in candidate next to the SVM. Training needs PyTorch
(requirements-dl.txt); the fitted network is exported to ONNX and inference
runs with OpenCV's DNN module, so the deployed app needs neither PyTorch nor
a GPU.

Architecture (~0.6 M parameters): 3 blocks of [conv3x3-BN-ReLU x2, maxpool],
32/64/128 channels, then dropout and two fully connected layers. Training
uses on-the-fly handwriting augmentation on each batch (rotation, shear,
anisotropic scale, translation, thinner/thicker pen), label smoothing and a
one-cycle learning-rate schedule. Classes are balanced by sampling weights.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from app.ml.symbol_features import CNN_SIZE, N_HANDCRAFTED

_IMG_SLICE = slice(N_HANDCRAFTED, N_HANDCRAFTED + CNN_SIZE * CNN_SIZE)


def _images(X: np.ndarray) -> np.ndarray:
    """(N,1,32,32) float images in [0, 1] from feature vectors, float images, or
    compact uint8 images (CNN_SIZE**2 columns, 0-255)."""
    X = np.asarray(X)
    if X.shape[1] == CNN_SIZE * CNN_SIZE:  # already raw images
        X = X.astype(np.float32) / 255.0 if X.dtype == np.uint8 else X.astype(np.float32)
        return X.reshape(-1, 1, CNN_SIZE, CNN_SIZE)
    return np.ascontiguousarray(X[:, _IMG_SLICE], dtype=np.float32).reshape(-1, 1, CNN_SIZE, CNN_SIZE)


def compact_images(X: np.ndarray) -> np.ndarray:
    """Feature vectors -> uint8 images (1 KB per symbol), for large training sets."""
    return np.round(_images(X).reshape(len(X), -1) * 255).astype(np.uint8)


def _build_net(n_classes: int):
    import torch.nn as nn

    def block(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2)]

    return nn.Sequential(*block(1, 32), *block(32, 64), *block(64, 128), nn.Flatten(),
                         nn.Dropout(0.3), nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
                         nn.Dropout(0.3), nn.Linear(256, n_classes))


def _augment_batch(x, gen):
    """Random affine warp + stroke-width change of a (B,1,32,32) batch."""
    import torch
    import torch.nn.functional as F

    b = x.shape[0]
    u = lambda lo, hi: torch.empty(b).uniform_(lo, hi, generator=gen)  # noqa: E731
    ang = u(-0.2, 0.2)  # radians, about +-11 degrees
    shear = u(-0.25, 0.25)
    sx, sy = u(0.85, 1.15), u(0.85, 1.15)
    tx, ty = u(-0.08, 0.08), u(-0.08, 0.08)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([cos * sx, -sin * sy + shear, tx], 1),
                         torch.stack([sin * sx, cos * sy, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    pen = torch.rand(b, generator=gen)
    thick = F.max_pool2d(x, 3, 1, 1)
    thin = -F.max_pool2d(-x, 3, 1, 1)
    x = torch.where((pen < 0.2).view(-1, 1, 1, 1), thick, x)
    x = torch.where((pen > 0.85).view(-1, 1, 1, 1), thin, x)
    return x


class CNNSymbolClassifier:
    def __init__(self, epochs: int = 18, batch_size: int = 256, lr: float = 3e-3, seed: int = 0, verbose: bool = False,
                 samples_per_epoch: int | None = None):
        self.epochs, self.batch_size, self.lr, self.seed, self.verbose = epochs, batch_size, lr, seed, verbose
        self.samples_per_epoch = samples_per_epoch  # None = one pass over the data

    # -- training (PyTorch) --------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None, log=print, init: "CNNSymbolClassifier | None" = None):
        """`init`: a fitted model with the same classes to fine-tune from (warm start)."""
        import io

        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        torch.set_num_threads(max(1, torch.get_num_threads()))
        gen = torch.Generator().manual_seed(self.seed)
        self.classes_ = np.array(sorted(set(y)))
        index = {c: i for i, c in enumerate(self.classes_)}
        X = np.asarray(X)
        u8 = X if X.dtype == np.uint8 and X.shape[1] == CNN_SIZE * CNN_SIZE else compact_images(X)
        imgs = torch.from_numpy(u8.reshape(-1, 1, CNN_SIZE, CNN_SIZE))
        labels = torch.tensor([index[c] for c in y])
        counts = np.bincount(labels.numpy(), minlength=len(self.classes_))
        weights = torch.from_numpy((1.0 / np.sqrt(counts))[labels.numpy()]).double()

        net = _build_net(len(self.classes_))
        if init is not None:
            if list(init.classes_) != list(self.classes_):
                raise ValueError("warm start needs the same classes")
            net.load_state_dict(torch.load(io.BytesIO(init.state_), weights_only=True))
        steps = math.ceil(min(len(labels), self.samples_per_epoch or len(labels)) / self.batch_size)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps * self.epochs)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
        t0 = time.perf_counter()
        for ep in range(self.epochs):
            net.train()
            order = torch.multinomial(weights, steps * self.batch_size, replacement=True, generator=gen)
            total = 0.0
            for s in range(steps):
                idx = order[s * self.batch_size:(s + 1) * self.batch_size]
                xb = _augment_batch(imgs[idx].float() / 255.0, gen)
                opt.zero_grad()
                loss = loss_fn(net(xb), labels[idx])
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss.detach()) * len(idx)
            if self.verbose:
                msg = f"epoch {ep + 1}/{self.epochs} loss {total / len(order):.4f} ({time.perf_counter() - t0:.0f}s)"
                if X_val is not None:
                    self._export(net)
                    msg += f" val acc {np.mean(self.predict(X_val) == np.asarray(y_val)):.4f}"
                log(msg)
        self._export(net)
        return self

    def _export(self, net) -> None:
        import io

        import torch

        net.eval()
        state = io.BytesIO()
        torch.save(net.state_dict(), state)
        self.state_ = state.getvalue()  # PyTorch weights, for warm-start fine-tuning
        buf = io.BytesIO()
        torch.onnx.export(net, torch.zeros(1, 1, CNN_SIZE, CNN_SIZE), buf, input_names=["glyph"],
                          output_names=["logits"], dynamic_axes={"glyph": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=13, dynamo=False)
        self.onnx_ = buf.getvalue()
        self._net = None

    # -- inference (OpenCV DNN) ----------------------------------------------
    def _dnn(self):
        if getattr(self, "_net", None) is None:
            self._net = cv2.dnn.readNetFromONNX(np.frombuffer(self.onnx_, np.uint8))
        return self._net

    def decision_function(self, X) -> np.ndarray:
        imgs = _images(X)
        out = []
        net = self._dnn()
        for s in range(0, len(imgs), 512):
            net.setInput(imgs[s:s + 512])
            out.append(net.forward().reshape(len(imgs[s:s + 512]), -1))
        return np.vstack(out) if out else np.zeros((0, len(self.classes_)), np.float32)

    def predict_proba(self, X) -> np.ndarray:
        z = self.decision_function(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_net"}
