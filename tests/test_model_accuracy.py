"""
Milestone 4 - Model Accuracy and Regression Tests

Checks that:
1. metrics.json exists and contains model accuracy values.
2. Every trained model meets the minimum accuracy threshold.
3. The expected models are present.
"""

import json
from pathlib import Path


ACCURACY_THRESHOLD = 75.0

METRICS_PATH = Path("model/metrics.json")


def load_metrics():
    """Load the CareerCast model metrics file."""
    assert METRICS_PATH.exists(), "model/metrics.json does not exist"

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_model_metrics_file_exists():
    """metrics.json must exist and contain valid JSON."""
    metrics = load_metrics()

    assert metrics is not None
    assert "models" in metrics
    assert isinstance(metrics["models"], dict)


def test_expected_models_are_present():
    """All three CareerCast models must have recorded metrics."""
    metrics = load_metrics()

    expected_models = {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }

    actual_models = set(metrics["models"].keys())

    assert expected_models.issubset(actual_models), (
        f"Missing model metrics. Expected: {expected_models}, "
        f"Found: {actual_models}"
    )


def test_accuracy_meets_threshold():
    """
    Accuracy gate.

    Every trained model must have accuracy >= 75%.
    If a future model update causes accuracy to fall below
    this threshold, the test will fail.
    """
    metrics = load_metrics()

    for model_name, model_metrics in metrics["models"].items():
        accuracy = model_metrics.get("accuracy")

        assert accuracy is not None, (
            f"{model_name} has no accuracy value in metrics.json"
        )

        assert accuracy >= ACCURACY_THRESHOLD, (
            f"{model_name} accuracy {accuracy}% is below the "
            f"required threshold of {ACCURACY_THRESHOLD}%"
        )


def test_accuracy_values_are_valid():
    """Accuracy values must be between 0 and 100."""
    metrics = load_metrics()

    for model_name, model_metrics in metrics["models"].items():
        accuracy = model_metrics.get("accuracy")

        assert 0 <= accuracy <= 100, (
            f"{model_name} has invalid accuracy: {accuracy}"
        )