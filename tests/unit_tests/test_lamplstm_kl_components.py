# Copyright 2026 The RLinf Authors.
"""Check KL decomposition against the distribution definition and target scaling."""

import numpy as np
import torch

from scripts.analyze_lamplstm_kl_components import kl_components


def test_matches_distribution_kl():
    mu = np.array([[0.0, 3.0], [0.4, -0.7]])
    lv = np.log(np.array([[1.0, 1.0], [0.01, 2.0]]))
    a, b = kl_components(mu, lv)
    q = torch.distributions.Normal(torch.tensor(mu), torch.tensor(np.exp(0.5 * lv)))
    p = torch.distributions.Normal(torch.zeros_like(q.loc), torch.ones_like(q.scale))
    expected = torch.distributions.kl_divergence(q, p).numpy()
    np.testing.assert_allclose(a + b, expected, atol=1e-12)
    assert a[0, 1] == 4.5 and b[0, 1] == 0


def test_posterior_width_can_change_kl_without_changing_dp_targets():
    mu = np.array([[-1.0, 2.0], [0.0, 0.0], [1.0, -2.0]])
    a, b = kl_components(mu, np.zeros_like(mu))
    c, d = kl_components(mu, np.full_like(mu, np.log(0.01)))
    np.testing.assert_array_equal(a, c)
    assert np.all(d > b)
    # The deterministic target mu is identical, even though posterior KL changed.
    np.testing.assert_array_equal(mu.mean(0), np.zeros(2))
