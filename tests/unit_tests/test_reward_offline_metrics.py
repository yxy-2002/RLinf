# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

from examples.reward.evaluate_ruiyan_reward import classification_metrics


def test_confusion_and_threshold_boundary():
    result = classification_metrics([1, 1, 0, 0], [0.9, 0.5, 0.8, 0.1], 0.5)
    assert [result[k] for k in ["tp", "fn", "fp", "tn"]] == [1, 1, 1, 1]
    assert result["accuracy"] == 0.5
    assert result["false_positive_rate"] == 0.5


def test_no_predicted_positives():
    result = classification_metrics([0, 1], [0.1, 0.2], 0.5)
    assert result["success_precision"] is None
    assert result["success_recall"] == 0
