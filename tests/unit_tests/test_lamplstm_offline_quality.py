# Copyright 2026 The RLinf Authors.
"""Regression checks for offline metric masks and train-only probes."""

import numpy as np

from scripts.analyze_lamplstm_offline_quality import grouped_cv, spearman
from scripts.eval_lamplstm_offline_quality import episode_ids, masked_mse, ridge_predict


def test_padding_cannot_change_error() -> None:
    target = np.zeros((2, 4, 3))
    prediction = np.ones_like(target)
    prediction[:, 2:] = 1e6
    mask = np.array([[1, 1, 0, 0], [1, 0, 0, 0]])
    np.testing.assert_allclose(masked_mse(prediction, target, mask), [1, 1])


def test_episode_boundaries_and_short_episode() -> None:
    future = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0]])
    history = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1]])
    np.testing.assert_array_equal(episode_ids(history, future), [0, 0, 1, 1, 1])


def test_ridge_has_no_validation_batch_dependence() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 3))
    y = (x[:, :2] * 0.4)[:, None, :]
    validation = rng.normal(size=(7, 3))
    whole = ridge_predict(x, y, validation)
    alone = ridge_predict(x, y, validation[:1])
    np.testing.assert_allclose(whole[:1], alone, atol=1e-7)
    assert np.mean((whole[:, 0] - validation[:, :2] * 0.4) ** 2) < 0.001


def test_group_cv_excludes_test_labels() -> None:
    x = np.arange(12, dtype=float)
    y = x / 20
    group = np.repeat(np.arange(3), 4)
    original = grouped_cv(x, y, group)
    y[group == 0] = 100
    changed = grouped_cv(x, y, group)
    np.testing.assert_array_equal(original[group == 0], changed[group == 0])
    np.testing.assert_allclose(spearman([1, 2, 2, 4], [4, 2, 2, 1]), -1.0)
