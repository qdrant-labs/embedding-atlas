# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Supervised linear probes on embedding vectors.

A probe measures how well a categorical attribute (e.g., a metadata column) can
be recovered from the original high-dimensional embedding vectors, by training a
regularized logistic regression classifier from the vectors to the attribute.

To keep every reported number honest, each labeled row is predicted by a model
that never saw it (stratified k-fold out-of-fold predictions). The per-row
correctness can therefore be aggregated over any subset of rows (e.g., a
selection in the embedding view) and still be a held-out estimate.

Folds are assigned from a hash of each vector rather than its row position, so
results don't depend on row order, and duplicate vectors always share a fold
(they can't leak between training and held-out rows).

- Core function: :func:`probe`
- DataFrame entry point: :func:`compute_probe_columns`
"""

import hashlib
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .utils import logger

CORRECT = "correct"
INCORRECT = "incorrect"
EXCLUDED = "excluded"


@dataclass
class ProbeResult:
    """Per-row outputs and evaluation report of a probe."""

    # Per-row arrays of length N (the number of input rows)

    # Predicted class (string), None for rows without a vector.
    predicted: np.ndarray
    # "correct" / "incorrect" for rows used in evaluation, "excluded" otherwise
    # (missing label, rare class, or missing vector).
    correct: np.ndarray
    # Probability of the predicted class, NaN for rows without a vector.
    confidence: np.ndarray

    # Evaluation report (all metrics are computed on out-of-fold predictions)

    classes: list[str]
    # Classes that were dropped for having too few labeled rows, with their counts.
    excluded_classes: dict[str, int]
    n_rows: int
    n_evaluated: int
    n_folds: int
    accuracy: float
    macro_f1: float
    # Accuracy of always predicting the training fold's majority class.
    baseline_accuracy: float
    # One dict per class: {class, precision, recall, f1, support}.
    per_class: list[dict] = field(default_factory=list)
    # Rows are true classes, columns are predicted classes, ordered as `classes`.
    confusion_matrix: list[list[int]] = field(default_factory=list)

    def report(self) -> dict:
        """Returns the evaluation report as a JSON-serializable dict."""
        return {
            "classes": self.classes,
            "excluded_classes": self.excluded_classes,
            "n_rows": self.n_rows,
            "n_evaluated": self.n_evaluated,
            "n_folds": self.n_folds,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "baseline_accuracy": self.baseline_accuracy,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
        }


def _normalize_labels(labels: pd.Series) -> np.ndarray:
    """Convert labels to an object array of strings, with None for missing labels."""
    values = labels
    if pd.api.types.is_float_dtype(values):
        finite = values.dropna()
        if not np.all(np.mod(finite, 1) == 0):
            raise ValueError(
                "the target looks continuous (non-integer numbers); "
                "probes only support categorical targets"
            )
        # Integral floats usually come from integer columns with missing values.
        values = values.astype("Int64")
    result = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            result[i] = None
        elif isinstance(v, (list, dict, np.ndarray)):
            raise ValueError("the target must contain scalar values, not lists")
        else:
            s = str(v)
            result[i] = s if s != "" else None
    return result


def _make_model(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # The scaler is part of the pipeline so it's fit on training rows only.
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))


def _vector_keys(vectors: np.ndarray, random_state: int) -> np.ndarray:
    """Seeded 64-bit hash of each vector's contents, as a uint64 array."""
    salt = random_state.to_bytes(8, "little", signed=True)
    digests = b"".join(
        hashlib.blake2b(v.tobytes(), digest_size=8, salt=salt).digest() for v in vectors
    )
    return np.frombuffer(digests, dtype=np.uint64)


def _assign_folds(keys: np.ndarray, y: np.ndarray, n_folds: int) -> np.ndarray:
    """
    Assign rows to folds from their vector keys, independently of row order.

    Rows with the same key (identical vectors) form a group that shares a fold.
    Groups are ordered by class, then by key (pseudo-random), and dealt to folds
    round-robin, so every class is spread evenly across folds.
    """
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    # A group with mixed labels is stratified by its smallest class index.
    group_class = np.full(len(unique_keys), np.iinfo(np.int64).max)
    np.minimum.at(group_class, inverse, y)
    order = np.lexsort((unique_keys, group_class))
    group_fold = np.empty(len(unique_keys), dtype=np.int64)
    group_fold[order] = np.arange(len(order)) % n_folds
    return group_fold[inverse]


def probe(
    vectors: np.ndarray,
    labels: pd.Series | list,
    *,
    has_vector: np.ndarray | None = None,
    n_folds: int = 5,
    min_class_count: int = 5,
    max_classes: int = 50,
    min_rows: int = 50,
    C: float = 1.0,
    random_state: int = 42,
) -> ProbeResult:
    """
    Train and evaluate a logistic regression probe from vectors to labels.

    Rows with a missing label, or whose class has fewer than ``min_class_count``
    distinct labeled vectors, are excluded from training and evaluation; they still
    receive a prediction from a model trained on all evaluated rows.

    Results are reproducible and independent of row order: folds are assigned
    from a seeded hash of each vector, and identical vectors share a fold.

    Args:
        vectors: Array of shape (N, D), the original embedding vectors.
        labels: N labels. Missing values (None, NaN, "") are treated as unlabeled.
        has_vector: Optional boolean mask of shape (N,). Rows where it is False
            are ignored (their entries in ``vectors`` can be arbitrary).
        n_folds: Number of stratified folds for out-of-fold predictions.
        min_class_count: Minimum number of distinct labeled vectors for a class
            to be kept (at least ``n_folds``).
        max_classes: Maximum number of (kept) classes.
        min_rows: Minimum number of evaluated rows.
        C: Inverse regularization strength of the logistic regression.
        random_state: Seed for the fold assignment.

    Returns:
        A :class:`ProbeResult`.

    Raises:
        ValueError: If the target is not a usable categorical attribute.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    vectors = np.asarray(vectors, dtype=np.float32)
    labels = _normalize_labels(pd.Series(labels).reset_index(drop=True))
    n = len(labels)
    if vectors.ndim != 2 or vectors.shape[0] != n:
        raise ValueError(
            f"expected vectors of shape ({n}, D), got {tuple(vectors.shape)}"
        )
    if has_vector is None:
        has_vector = np.ones(n, dtype=bool)

    labeled = has_vector & np.array([v is not None for v in labels])
    keys = np.zeros(n, dtype=np.uint64)
    keys[labeled] = _vector_keys(vectors[labeled], random_state)
    labeled_rows = pd.DataFrame({"label": labels[labeled], "key": keys[labeled]})
    counts = labeled_rows["label"].value_counts()
    # Duplicate vectors share a fold, so count distinct vectors per class.
    distinct_counts = labeled_rows.groupby("label")["key"].nunique()
    min_class_count = max(min_class_count, n_folds)
    classes = sorted(c for c, k in distinct_counts.items() if k >= min_class_count)
    excluded_classes = {
        str(c): int(counts[c])
        for c, k in distinct_counts.items()
        if k < min_class_count
    }
    if len(classes) < 2:
        raise ValueError(
            f"need at least 2 classes with {min_class_count} or more labeled rows, "
            f"found {len(classes)}"
        )
    if len(classes) > max_classes:
        raise ValueError(
            f"too many classes ({len(classes)}, the maximum is {max_classes}); "
            "probes are meant for categorical attributes"
        )

    class_index = {c: i for i, c in enumerate(classes)}
    evaluated = labeled & np.array([v in class_index for v in labels])
    n_evaluated = int(evaluated.sum())
    if n_evaluated < min_rows:
        raise ValueError(f"need at least {min_rows} labeled rows, found {n_evaluated}")

    # Train on the evaluated rows in a canonical order (by vector key, then class),
    # since the solver's floating-point results depend slightly on row order.
    rows = np.flatnonzero(evaluated)
    row_classes = np.array([class_index[v] for v in labels[rows]])
    canonical = np.lexsort((row_classes, keys[rows]))
    rows, row_classes = rows[canonical], row_classes[canonical]
    x = vectors[rows]
    y = row_classes
    k = len(classes)

    # Out-of-fold predictions: every evaluated row is predicted by a model
    # trained on the other folds.
    proba = np.zeros((n_evaluated, k), dtype=np.float64)
    baseline_hits = 0
    folds = _assign_folds(keys[rows], y, n_folds)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        for fold in range(n_folds):
            test = folds == fold
            train = ~test
            if not test.any():
                continue
            model = _make_model(C).fit(x[train], y[train])
            # Map the model's classes to columns, in case a class is missing from
            # this training fold (only possible with mixed-label duplicates).
            proba[np.ix_(test, model.classes_)] = model.predict_proba(x[test])
            majority = np.bincount(y[train], minlength=k).argmax()
            baseline_hits += int(np.sum(y[test] == majority))

        # Rows not used for evaluation are predicted by a model fit on all evaluated rows.
        others = has_vector & ~evaluated
        others_proba = None
        if others.any():
            model = _make_model(C).fit(x, y)
            others_proba = model.predict_proba(vectors[others])
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        logger.warning(
            "Probe: logistic regression did not fully converge; "
            "results may be slightly suboptimal."
        )

    y_pred = proba.argmax(axis=1)
    class_names = np.array(classes, dtype=object)

    predicted = np.full(n, None, dtype=object)
    confidence = np.full(n, np.nan, dtype=np.float64)
    correct = np.full(n, EXCLUDED, dtype=object)
    predicted[rows] = class_names[y_pred]
    confidence[rows] = proba.max(axis=1)
    correct[rows] = np.where(y_pred == y, CORRECT, INCORRECT)
    if others_proba is not None:
        predicted[others] = class_names[others_proba.argmax(axis=1)]
        confidence[others] = others_proba.max(axis=1)

    labels_range = list(range(k))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=labels_range, zero_division=0
    )
    per_class = [
        {
            "class": classes[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels_range
    ]

    return ProbeResult(
        predicted=predicted,
        correct=correct,
        confidence=confidence,
        classes=classes,
        excluded_classes=excluded_classes,
        n_rows=n,
        n_evaluated=n_evaluated,
        n_folds=n_folds,
        accuracy=float(accuracy_score(y, y_pred)),
        macro_f1=float(
            f1_score(y, y_pred, labels=labels_range, average="macro", zero_division=0)
        ),
        baseline_accuracy=baseline_hits / n_evaluated,
        per_class=per_class,
        confusion_matrix=confusion_matrix(y, y_pred, labels=labels_range).tolist(),
    )


def compute_probe_columns(
    dataframe: pd.DataFrame,
    *,
    vector: str,
    target: str,
    **kwargs,
) -> ProbeResult:
    """
    Train a probe from a DataFrame's vector column to its target column.

    Args:
        dataframe: pandas DataFrame containing the data.
        vector: Column name containing the embedding vectors (one list or array per row).
            Rows with a missing vector are excluded.
        target: Column name of the categorical attribute to predict.
        **kwargs: Additional arguments for :func:`probe`.

    Returns:
        A :class:`ProbeResult`, whose per-row arrays align with the DataFrame's rows.
    """
    if vector not in dataframe.columns:
        raise ValueError(f"vector column {vector!r} not found")
    if target not in dataframe.columns:
        raise ValueError(f"target column {target!r} not found")
    if target == vector:
        raise ValueError("the target column cannot be the vector column")

    values = dataframe[vector].to_list()
    has_vector = np.array([v is not None for v in values])
    if not has_vector.any():
        raise ValueError(f"vector column {vector!r} has no vectors")
    dims = {len(v) for v, ok in zip(values, has_vector) if ok}
    if len(dims) != 1:
        raise ValueError(f"vectors in column {vector!r} have inconsistent lengths")
    (dim,) = dims
    vectors = np.zeros((len(values), dim), dtype=np.float32)
    for i, v in enumerate(values):
        if has_vector[i]:
            vectors[i] = np.asarray(v, dtype=np.float32)

    return probe(vectors, dataframe[target], has_vector=has_vector, **kwargs)


def format_probe_report(result: ProbeResult, target: str) -> str:
    """Format a probe's evaluation report as human-readable text."""
    lines = [
        (
            f"Probe for {target!r} ({result.n_evaluated} rows, "
            f"{len(result.classes)} classes, {result.n_folds}-fold out-of-fold predictions):"
        ),
        (
            f"  accuracy {result.accuracy:.3f} · macro-F1 {result.macro_f1:.3f} · "
            f"majority baseline {result.baseline_accuracy:.3f}"
        ),
    ]
    if result.excluded_classes:
        lines.append(
            f"  excluded {len(result.excluded_classes)} rare class(es): "
            + ", ".join(f"{c} ({n})" for c, n in result.excluded_classes.items())
        )
    width = max(len(c["class"]) for c in result.per_class)
    lines.append(f"  {'class':<{width}}  precision  recall  support")
    for c in result.per_class:
        lines.append(
            f"  {c['class']:<{width}}  {c['precision']:>9.3f}  {c['recall']:>6.3f}  {c['support']:>7d}"
        )
    return "\n".join(lines)


def probe_confusion_chart(
    result: ProbeResult, *, target: str, predicted: str, target_is_string: bool
) -> dict:
    """
    Returns a heatmap chart spec (for the frontend's ``defaultChartsConfig``)
    showing the confusion matrix between the target and predicted columns.
    """

    def quote(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    # Cast non-string targets so the heatmap treats them as categories, not bins.
    x_field = (
        target if target_is_string else {"sql": f"CAST({quote(target)} AS VARCHAR)"}
    )
    return {
        # Most important numbers first, as long titles are truncated in the UI.
        "title": (
            f"Probe {target}: accuracy {result.accuracy:.2f} "
            f"(baseline {result.baseline_accuracy:.2f}) · macro-F1 {result.macro_f1:.2f}"
        ),
        "layers": [
            {
                "mark": "rect",
                "filter": "$filter",
                "zIndex": -1,
                "encoding": {
                    "x": {"field": x_field},
                    "y": {"field": predicted},
                    "color": {"aggregate": "count"},
                },
            },
            {"mark": "rect", "zIndex": -2, "encoding": {"color": {"value": 0}}},
        ],
        "selection": {"brush": {"encoding": "xy"}},
        "widgets": [
            {
                "type": "encoding.normalize",
                "attribute": "color",
                "layer": 0,
                "options": ["x", "y"],
            },
        ],
    }
