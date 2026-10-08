"""Rasch (1PL IRT) by regularized joint maximum likelihood, plus an
explanatory difficulty model (ridge regression on item features) used for
items with too few responses."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score


def fit_rasch(learner_idx: np.ndarray, item_idx: np.ndarray, y: np.ndarray, n_learners: int, n_items: int,
              l2: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Returns (theta, b). P(correct) = sigmoid(theta_j - b_i)."""

    def nll(params):
        theta, b = params[:n_learners], params[n_learners:]
        z = theta[learner_idx] - b[item_idx]
        p = 1 / (1 + np.exp(-z))
        ll = y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)
        r = y - p
        g_theta = -np.bincount(learner_idx, weights=r, minlength=n_learners) + 2 * l2 * theta
        g_b = np.bincount(item_idx, weights=r, minlength=n_items) + 2 * l2 * b
        return -ll.sum() + l2 * (theta @ theta + b @ b), np.concatenate([g_theta, g_b])

    x0 = np.zeros(n_learners + n_items)
    res = minimize(nll, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    theta, b = res.x[:n_learners], res.x[n_learners:]
    shift = b.mean()
    return theta - shift, b - shift


def fit_difficulty_model(X: np.ndarray, b: np.ndarray, seed: int = 0) -> tuple[Ridge, float]:
    model = Ridge(alpha=1.0)
    r2 = float(cross_val_score(model, X, b, cv=KFold(5, shuffle=True, random_state=seed), scoring="r2").mean())
    return model.fit(X, b), r2
