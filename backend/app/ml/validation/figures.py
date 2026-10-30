"""Paper figure from the validation reports.

    cd backend && python -m app.ml.validation.figures     # writes docs/fig2.pdf

Fig. 2: diagnosis accuracy against the number of symbols a student is asked
to check per page, as the confirm-before-diagnose threshold tau rises.
Each marker is one tau (0.30 ... 0.98, left to right); x = 0 is no
confirmation. Palette: categorical slots 1-3 of the dataviz reference palette
(validated all-pairs in light mode); distinct markers carry identity too.
"""
from __future__ import annotations

import json

from app.ml.artifacts import BACKEND_ROOT
from app.ml.model_cards import REPORT_DIR

SERIES = [("clean", "Clean page", "#2a78d6", "o"), ("phone_mild", "Phone photo, mild", "#eb6834", "s"),
          ("phone_moderate", "Phone photo, moderate", "#1baf7a", "^")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def fig_confirmation_tradeoff(out=BACKEND_ROOT.parent / "docs" / "fig2.pdf") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = json.loads((REPORT_DIR / "validation_propagation.json").read_text())
    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ref = 100 * rep["reference_accuracy_true_transcription"]
    ax.axhline(ref, color=MUTED, linestyle="--", linewidth=0.9)
    for key, label, color, marker in SERIES:
        base = rep["conditions"][key]["no_confirmation"]
        pts = [(0.0, 100 * base["accuracy"])] + [(c["symbols_to_check_per_page"], 100 * c["accuracy"])
                                                 for c in rep["tau_curves"][key]]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=1.6, marker=marker, markersize=3.2, label=label,
                markeredgecolor="white", markeredgewidth=0.4)
    chosen = next(c for c in rep["tau_curves"]["clean"] if c["tau"] == rep["tau"])
    ax.annotate(f"deployed $\\tau$ = {rep['tau']:g}", (chosen["symbols_to_check_per_page"], 100 * chosen["accuracy"]),
                xytext=(chosen["symbols_to_check_per_page"] - 0.4, 58), ha="center", fontsize=6.5, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.7})
    ax.set_xlabel("Symbols the student checks per page")
    ax.set_ylabel("Diagnosis accuracy (%)")
    ax.set_ylim(0, 100)
    right = max(c["symbols_to_check_per_page"] for k, *_ in SERIES for c in rep["tau_curves"][k]) * 1.08
    ax.set_xlim(left=-0.3, right=right)
    ax.text(0.05, ref - 2.5, f"diagnosed from true transcription: {ref:.1f}%", color=MUTED, fontsize=6, va="top")
    ax.legend(loc="lower right", fontsize=6.5, frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_confirmation_tradeoff()
