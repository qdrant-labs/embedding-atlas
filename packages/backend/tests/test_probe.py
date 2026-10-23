# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Tests for supervised linear probes on embedding vectors.

Covers held-out evaluation (including a leakage check on random labels),
handling of missing labels, rare classes, missing vectors and invalid targets,
reproducibility, and the compute_probe_columns DataFrame entry point.
"""

import numpy as np
import pandas as pd
import pytest
from embedding_atlas.options import make_embedding_atlas_props
from embedding_atlas.probe import (
    compute_probe_columns,
    format_probe_report,
    probe,
    probe_confusion_chart,
)


def make_blobs(n_per_class=100, classes=("a", "b", "c"), dim=16, spread=0.5, seed=0):
    """Well-separated Gaussian clusters, one per class."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(classes), dim)) * 5
    vectors = np.concatenate(
        [c + rng.normal(scale=spread, size=(n_per_class, dim)) for c in centers]
    )
    labels = np.repeat(np.array(classes, dtype=object), n_per_class)
    return vectors, labels


class TestEvaluation:
    def test_separable_classes(self):
        vectors, labels = make_blobs()
        result = probe(vectors, labels)
        assert result.classes == ["a", "b", "c"]
        assert result.n_evaluated == 300
        assert result.accuracy > 0.95
        assert result.macro_f1 > 0.95
        assert result.baseline_accuracy < 0.5
        assert np.all(result.correct == "correct")
        assert np.all((result.confidence > 0) & (result.confidence <= 1))

    def test_random_labels_score_near_chance(self):
        """With more dimensions than rows, a model scored on its own training rows
        would be near-perfect; out-of-fold predictions must stay near chance."""
        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(200, 400))
        labels = np.array(["x", "y"] * 100, dtype=object)
        result = probe(vectors, labels)
        assert result.accuracy < 0.7

    def test_majority_baseline_on_imbalanced_classes(self):
        vectors, labels = make_blobs(n_per_class=100, classes=("a", "b"))
        keep = np.r_[0:100, 100:125]
        result = probe(vectors[keep], labels[keep])
        assert result.baseline_accuracy == pytest.approx(100 / 125)

    def test_confusion_matrix_and_per_class(self):
        vectors, labels = make_blobs(spread=6.0)
        result = probe(vectors, labels)
        matrix = np.array(result.confusion_matrix)
        assert matrix.shape == (3, 3)
        assert matrix.sum() == result.n_evaluated
        assert np.trace(matrix) / matrix.sum() == pytest.approx(result.accuracy)
        assert [c["class"] for c in result.per_class] == result.classes
        assert [c["support"] for c in result.per_class] == list(matrix.sum(axis=1))
        n_correct = int(np.sum(result.correct == "correct"))
        assert n_correct == np.trace(matrix)

    def test_reproducible(self):
        vectors, labels = make_blobs(spread=6.0)
        a = probe(vectors, labels)
        b = probe(vectors, labels)
        assert a.report() == b.report()
        assert np.array_equal(a.predicted, b.predicted)
        assert np.array_equal(a.confidence, b.confidence)

    def test_independent_of_row_order(self):
        vectors, labels = make_blobs(spread=6.0)
        a = probe(vectors, labels)
        perm = np.random.default_rng(3).permutation(len(labels))
        b = probe(vectors[perm], labels[perm])
        assert a.report() == b.report()
        assert np.array_equal(a.predicted[perm], b.predicted)
        assert np.array_equal(a.correct[perm], b.correct)
        assert np.array_equal(a.confidence[perm], b.confidence)

    def test_seed_changes_folds(self):
        vectors, labels = make_blobs(spread=6.0)
        a = probe(vectors, labels, random_state=1)
        b = probe(vectors, labels, random_state=2)
        assert not np.array_equal(a.confidence, b.confidence)

    def test_duplicates_do_not_leak_between_folds(self):
        """Each vector appears twice with the same random label. If copies were
        split across folds, the model would recall the label of the training copy
        and score near 100%; with duplicates grouped, it stays near chance."""
        rng = np.random.default_rng(4)
        unique = rng.normal(size=(150, 400))
        unique_labels = rng.choice(np.array(["x", "y"], dtype=object), size=150)
        vectors = np.concatenate([unique, unique])
        labels = np.concatenate([unique_labels, unique_labels])
        result = probe(vectors, labels)
        assert result.n_evaluated == 300
        assert result.accuracy < 0.7

    def test_metrics_match_per_row_outputs(self):
        vectors, labels = make_blobs(spread=6.0)
        result = probe(vectors, labels)
        assert np.mean(result.predicted == labels) == pytest.approx(result.accuracy)


class TestLabels:
    def test_missing_labels_are_excluded_but_predicted(self):
        vectors, labels = make_blobs()
        labels = labels.copy()
        labels[[0, 1, 2]] = [None, np.nan, ""]
        result = probe(vectors, labels)
        assert result.n_evaluated == 297
        assert list(result.correct[:3]) == ["excluded"] * 3
        assert list(result.predicted[:3]) == ["a"] * 3
        assert not np.any(np.isnan(result.confidence))

    def test_rare_classes_are_excluded(self):
        vectors, labels = make_blobs()
        labels = labels.copy()
        labels[:3] = "rare"
        result = probe(vectors, labels)
        assert result.classes == ["a", "b", "c"]
        assert result.excluded_classes == {"rare": 3}
        assert list(result.correct[:3]) == ["excluded"] * 3
        assert "rare" not in set(result.predicted)

    def test_rare_class_counts_distinct_vectors(self):
        """A class with enough rows but too few distinct vectors is excluded."""
        vectors, labels = make_blobs()
        vectors, labels = vectors.copy(), labels.copy()
        vectors[:8] = vectors[0]
        vectors[8:10] = vectors[8]
        labels[:10] = "dup"
        result = probe(vectors, labels)
        assert result.excluded_classes == {"dup": 10}
        assert "dup" not in result.classes

    def test_integer_labels(self):
        vectors, _ = make_blobs(classes=(0, 1, 2))
        labels = pd.Series(np.repeat([3, 1, 2], 100))
        result = probe(vectors, labels)
        assert result.classes == ["1", "2", "3"]

    def test_integral_floats_with_missing_values(self):
        vectors, _ = make_blobs(classes=(0, 1))
        labels = pd.Series(np.repeat([1.0, 2.0], 100))
        labels[0] = np.nan
        result = probe(vectors, labels)
        assert result.classes == ["1", "2"]
        assert result.correct[0] == "excluded"

    def test_continuous_target_raises(self):
        vectors, _ = make_blobs()
        labels = np.random.default_rng(0).normal(size=300)
        with pytest.raises(ValueError, match="continuous"):
            probe(vectors, labels)

    def test_single_class_raises(self):
        vectors, _ = make_blobs()
        with pytest.raises(ValueError, match="at least 2 classes"):
            probe(vectors, ["a"] * 300)

    def test_too_many_classes_raises(self):
        vectors, _ = make_blobs()
        labels = [f"id{i}" for i in range(60)] * 5
        with pytest.raises(ValueError, match="too many classes"):
            probe(vectors, labels)

    def test_too_few_rows_raises(self):
        vectors, labels = make_blobs(n_per_class=10)
        with pytest.raises(ValueError, match="at least 50"):
            probe(vectors, labels)

    def test_list_labels_raise(self):
        vectors, _ = make_blobs()
        with pytest.raises(ValueError, match="scalar"):
            probe(vectors, pd.Series([["a"]] * 300))

    def test_vector_shape_mismatch_raises(self):
        vectors, labels = make_blobs()
        with pytest.raises(ValueError, match="shape"):
            probe(vectors[:10], labels)


class TestDataFrame:
    def make_df(self):
        vectors, labels = make_blobs()
        return pd.DataFrame({"vec": list(vectors), "label": labels})

    def test_columns_align_with_rows(self):
        df = self.make_df().sample(frac=1, random_state=0)
        result = compute_probe_columns(df, vector="vec", target="label")
        assert len(result.predicted) == len(df)
        assert np.all(result.predicted == df["label"].to_numpy())

    def test_list_vectors_and_missing_vectors(self):
        df = self.make_df()
        df["vec"] = [list(v) for v in df["vec"]]
        df.loc[0, "vec"] = None
        result = compute_probe_columns(df, vector="vec", target="label")
        assert result.predicted[0] is None
        assert np.isnan(result.confidence[0])
        assert result.correct[0] == "excluded"
        assert result.n_evaluated == 299

    def test_does_not_modify_dataframe(self):
        df = self.make_df()
        before = df.copy()
        compute_probe_columns(df, vector="vec", target="label")
        assert list(df.columns) == list(before.columns)
        assert df["label"].equals(before["label"])

    def test_invalid_columns_raise(self):
        df = self.make_df()
        with pytest.raises(ValueError, match="not found"):
            compute_probe_columns(df, vector="missing", target="label")
        with pytest.raises(ValueError, match="not found"):
            compute_probe_columns(df, vector="vec", target="missing")
        with pytest.raises(ValueError, match="cannot be the vector"):
            compute_probe_columns(df, vector="vec", target="vec")

    def test_inconsistent_vector_lengths_raise(self):
        df = self.make_df()
        df["vec"] = [list(v) for v in df["vec"]]
        df.at[0, "vec"] = [1.0, 2.0]
        with pytest.raises(ValueError, match="inconsistent"):
            compute_probe_columns(df, vector="vec", target="label")


class TestOutputs:
    def test_format_report(self):
        vectors, labels = make_blobs()
        labels = labels.copy()
        labels[:3] = "rare"
        text = format_probe_report(probe(vectors, labels), "label")
        assert "accuracy" in text and "majority baseline" in text
        assert "rare (3)" in text

    def test_confusion_chart(self):
        vectors, labels = make_blobs()
        result = probe(vectors, labels)
        chart = probe_confusion_chart(
            result, target="label", predicted="pred", target_is_string=True
        )
        encoding = chart["layers"][0]["encoding"]
        assert encoding["x"] == {"field": "label"}
        assert encoding["y"] == {"field": "pred"}
        assert chart["title"].startswith("Probe label: accuracy 1.00 (baseline 0.33)")

        chart = probe_confusion_chart(
            result, target='my "rating"', predicted="pred", target_is_string=False
        )
        x = chart["layers"][0]["encoding"]["x"]["field"]
        assert x == {"sql": 'CAST("my ""rating""" AS VARCHAR)'}

    def test_default_charts_config_prop(self):
        props = make_embedding_atlas_props(
            default_charts_config={"override": {"pred": {"title": "x"}}}
        )
        assert props["defaultChartsConfig"] == {"override": {"pred": {"title": "x"}}}
