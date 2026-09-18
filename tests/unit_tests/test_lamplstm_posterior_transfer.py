# Copyright 2026 The RLinf Authors.
"""Check posterior-relative errors and fair masked perturbation controls."""

import numpy as np
import pytest

from scripts.analyze_lamplstm_posterior_transfer import (
    cluster_interval,
    lag_alignment,
    match_window_energy,
    numerical_agreement,
    ratio_summary,
)


def test_numerical_agreement_accepts_roundoff_but_rejects_changed_outputs():
    cached = np.zeros((2, 8, 2))
    result = numerical_agreement(cached + 1e-4, cached)
    assert result["rms"] == pytest.approx(1e-4)
    with pytest.raises(ValueError, match="differs materially"):
        numerical_agreement(cached + 0.02, cached)


def test_energy_matches_in_dp_coordinates_ignoring_padding_and_unused_tail():
    rng = np.random.default_rng(5)
    delta = rng.normal(size=(3, 16, 2))
    noise = rng.normal(size=delta.shape)
    scale = np.array([0.01, 9.0])
    mask = np.ones((3, 16))
    mask[0, 3:] = 0
    result = match_window_energy(noise, delta, mask, scale)
    for i in range(3):
        valid = mask[i, :8] > 0
        np.testing.assert_allclose(
            np.sum((result[i, :8][valid] / scale) ** 2),
            np.sum((delta[i, :8][valid] / scale) ** 2),
        )
    contaminated = delta.copy()
    contaminated[0, 3:] = 1e10
    contaminated[:, 8:] = -1e10
    np.testing.assert_allclose(
        match_window_energy(noise, contaminated, mask, scale), result
    )


def test_zero_error_produces_zero_matched_perturbation():
    shape = (2, 16, 2)
    np.testing.assert_array_equal(
        match_window_energy(
            np.ones(shape), np.zeros(shape), np.ones(shape[:2]), np.ones(2)
        ),
        np.zeros(shape),
    )


def test_sigma_units_are_invariant_under_affine_latent_rescaling():
    mu = np.array([[[0.2, 0.6], [0.1, -0.3]]])
    sigma = np.array([[[0.1, 0.2], [0.4, 0.3]]])
    pred = mu[None] + sigma[None] * np.array([[[[1, -4], [2, 3]]]])
    a, b = np.array([0.01, 7]), np.array([8, -3])
    ratio = (pred - mu) / sigma
    rescaled = ((pred * a + b) - (mu * a + b)) / (sigma * a)
    np.testing.assert_allclose(ratio, rescaled, atol=1e-10)
    summary = ratio_summary(ratio, np.array([[1, 0]]))
    assert summary["fraction_abs_gt3"] == 0.5
    np.testing.assert_allclose(summary["rms"], np.sqrt(8.5))


def test_episode_bootstrap_preserves_window_weighted_mean_and_pairing():
    values = np.array([1.0, 1.0, 1.0, 5.0])
    episodes = np.array([0, 0, 0, 1])
    result = cluster_interval(values, episodes)
    assert result["mean"] == 2.0
    assert result["per_episode"] == [1.0, 5.0]
    assert result["episode_bootstrap_ci95"] == [1.0, 5.0]
    paired = cluster_interval(values - values, episodes)
    assert paired["episode_bootstrap_ci95"] == [0.0, 0.0]


def test_temporal_alignment_distinguishes_coherent_and_alternating_errors():
    same = np.ones((1, 8, 2))
    alternate = same.copy()
    alternate[:, ::2] *= -1
    mask = np.ones((1, 8))
    assert lag_alignment(same, mask, np.ones(2)) == 1.0
    assert lag_alignment(alternate, mask, np.ones(2)) == -1.0
