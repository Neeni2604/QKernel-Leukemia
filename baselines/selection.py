"""selection.py - fold-safe gene selection for QKernel Leukemia.

WelchTSelector is the same Welch t-test used in the exploratory t-test script,
but packaged as a scikit-learn transformer. Inside a Pipeline it is re-fit on the
TRAINING patients of every CV fold, so test patients never influence which genes
are chosen (no leakage).

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("select", WelchTSelector(k=4)), ("scale", StandardScaler()), ("clf", SVC())])

Later, the QAOA step can reuse `prefilter_k` here: take the top ~16-20 genes by
Welch t-score, then choose the final 4-8 with relevance/redundancy QUBO.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin


def welch_t_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|Welch t| for every column of X, comparing class 0 vs class 1. NaN -> 0."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    with np.errstate(all="ignore"):
        t, _ = ttest_ind(X[y == 0], X[y == 1], axis=0, equal_var=False)
    return np.nan_to_num(np.abs(t), nan=0.0, posinf=0.0, neginf=0.0)


class WelchTSelector(BaseEstimator, TransformerMixin):
    """Keep the k columns with the largest |Welch t| between the two classes."""

    def __init__(self, k: int = 4):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.scores_ = welch_t_scores(X, y)
        k = min(self.k, X.shape[1])
        # stable ordering: highest score first, ties broken by column index
        order = np.lexsort((np.arange(len(self.scores_)), -self.scores_))
        self.selected_idx_ = np.sort(order[:k])
        self.ranked_idx_ = order  # full ranking, best first
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.selected_idx_]

    def get_support(self, indices: bool = False):
        if indices:
            return self.selected_idx_
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_idx_] = True
        return mask