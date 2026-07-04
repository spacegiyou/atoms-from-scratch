"""
Tests that check the physics is actually correct.

These are the same sanity checks you reuse on every *learned* potential in
later notebooks, so they are worth understanding, not just running:

  * a potential's force must equal the negative numerical gradient of its
    energy (test_force_matches_numerical_gradient), and
  * an isolated system integrated with velocity-Verlet must conserve total
    energy (test_energy_is_conserved).

Run with:  pytest -q
"""

import numpy as np
import pytest

from afs.md import (
    lj_energy,
    lj_forces,
    run_md,
    triangular_lattice,
    thermal_velocities,
)


def test_lj_minimum_is_at_expected_distance_and_depth():
    """Two atoms at r = 2**(1/6) should sit exactly at the bottom of the well."""
    r_min = 2.0 ** (1.0 / 6.0)
    pos = np.array([[0.0, 0.0], [r_min, 0.0]])
    assert np.isclose(lj_energy(pos), -1.0, atol=1e-9)
    # ...and the force there should vanish (bottom of the well is flat).
    assert np.allclose(lj_forces(pos), 0.0, atol=1e-8)


def test_force_matches_numerical_gradient():
    """The analytic force must equal -dV/dx from finite differences.

    This is the single most important check in the whole repo: it is how you
    verify that *any* potential's forces are consistent with its energy.
    """
    rng = np.random.default_rng(1)
    pos = triangular_lattice(3, 3) + 0.05 * rng.standard_normal((9, 2))

    analytic = lj_forces(pos)

    h = 1e-6
    numerical = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for k in range(pos.shape[1]):
            p_plus, p_minus = pos.copy(), pos.copy()
            p_plus[i, k] += h
            p_minus[i, k] -= h
            numerical[i, k] = -(lj_energy(p_plus) - lj_energy(p_minus)) / (2 * h)

    assert np.allclose(analytic, numerical, atol=1e-4)


def test_energy_is_conserved():
    """Velocity-Verlet on an isolated cluster should conserve total energy."""
    pos = triangular_lattice(4, 4)
    vel = thermal_velocities(len(pos), d=2, temp=0.2, seed=0)

    out = run_md(pos, vel, dt=0.002, steps=3000, record_every=10)
    total = out["total"]

    relative_drift = (total.max() - total.min()) / abs(total.mean())
    assert relative_drift < 0.02, f"energy drifted by {relative_drift:.3%}"


def test_thermal_velocities_have_zero_net_momentum():
    """A drifting cluster would be a bug; total momentum must be ~0."""
    vel = thermal_velocities(16, d=2, temp=0.3, seed=3)
    assert np.allclose(vel.sum(axis=0), 0.0, atol=1e-10)


def test_thermal_velocities_hit_target_temperature():
    """The generated velocities should match the requested temperature."""
    from afs.md import temperature
    vel = thermal_velocities(64, d=2, temp=0.35, seed=7)
    assert np.isclose(temperature(vel), 0.35, atol=1e-6)


def test_thermal_velocities_reject_too_few_atoms():
    """With fewer than two atoms, zero-net-momentum leaves nothing to heat,
    so the function should fail loudly instead of returning NaNs."""
    for n in (0, 1):
        with pytest.raises(ValueError):
            thermal_velocities(n, d=2, temp=0.3)
