"""
Molecular dynamics from scratch — in about 60 lines of NumPy.

Everything here is deliberately simple and explicit. There is no hidden
magic. The only physics you need:

    * energy is a number that depends on where the atoms are,
    * force is the negative gradient of that energy,  F = -dV/dx,
    * and atoms move by  F = m a,  integrated one small step at a time.

We use Lennard-Jones *reduced units*, where the natural scales are set to 1:
epsilon = sigma = mass = k_B = 1.  This keeps every formula clean and lets a
single dt work for any system. Real units are just a rescaling on top.

These same functions are reproduced (and explained line by line) in
notebooks/01_molecular_dynamics_from_scratch.ipynb. This module is the
version the test-suite checks, so the notebook can stay trustworthy.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# 1. The energy: the Lennard-Jones potential
# --------------------------------------------------------------------------
def lj_energy(positions: np.ndarray, epsilon: float = 1.0, sigma: float = 1.0) -> float:
    """Total Lennard-Jones potential energy of a set of atoms.

    For every unique pair of atoms at distance r,
        V(r) = 4 * epsilon * [ (sigma/r)**12 - (sigma/r)**6 ].
    The r**-12 term is a steep repulsion (atoms cannot overlap); the r**-6
    term is a gentle attraction (atoms like to stick together). The balance
    sits at r = 2**(1/6) * sigma, the bottom of the well, with energy -epsilon.

    positions : (N, d) array of atomic coordinates.
    """
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):        # each pair once
            rij = positions[i] - positions[j]
            r2 = rij @ rij
            sr2 = sigma * sigma / r2
            sr6 = sr2 ** 3
            sr12 = sr6 ** 2
            energy += 4.0 * epsilon * (sr12 - sr6)
    return energy


# --------------------------------------------------------------------------
# 2. The force: the analytic gradient of the energy
# --------------------------------------------------------------------------
def lj_forces(positions: np.ndarray, epsilon: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Force on every atom, F_i = -dV/d(r_i), as an (N, d) array.

    Differentiating the pair potential by hand gives, for the pair (i, j),
        F_i = 24 * epsilon / r**2 * [ 2*(sigma/r)**12 - (sigma/r)**6 ] * (r_i - r_j),
    and by Newton's third law F_j = -F_i. We accumulate both at once.

    Notebook 03 replaces this hand-derived gradient with autograd on a neural
    network. Notebook 01's test suite checks this analytic force against a
    finite-difference gradient, which is the same sanity check you will reuse
    on every learned potential later.
    """
    n = len(positions)
    forces = np.zeros_like(positions, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r2 = rij @ rij
            sr2 = sigma * sigma / r2
            sr6 = sr2 ** 3
            sr12 = sr6 ** 2
            f = 24.0 * epsilon * (2.0 * sr12 - sr6) / r2
            forces[i] += f * rij
            forces[j] -= f * rij
    return forces


# --------------------------------------------------------------------------
# 3. Bookkeeping: kinetic energy and temperature
# --------------------------------------------------------------------------
def kinetic_energy(velocities: np.ndarray, mass: float = 1.0) -> float:
    """Total kinetic energy, (1/2) m sum(v**2)."""
    return 0.5 * mass * float(np.sum(velocities ** 2))


def temperature(velocities: np.ndarray, mass: float = 1.0) -> float:
    """Instantaneous kinetic temperature (with k_B = 1).

    Equipartition says each degree of freedom carries (1/2) k_B T of kinetic
    energy, so T = 2 * KE / (d * N). We ignore the center-of-mass constraint
    for simplicity; accounting for it would use (d*N - d) in the denominator.
    """
    n, d = velocities.shape
    return 2.0 * kinetic_energy(velocities, mass) / (d * n)


# --------------------------------------------------------------------------
# 4. The integrator: one velocity-Verlet step
# --------------------------------------------------------------------------
def velocity_verlet_step(positions, velocities, forces, dt,
                         mass=1.0, epsilon=1.0, sigma=1.0):
    """Advance the system by one time step and return the new state.

    The whole integrator is four lines of physics:
        v_half = v + 0.5 * a * dt          # half-kick
        x'     = x + v_half * dt           # drift
        a'     = F(x') / m                 # new force
        v'     = v_half + 0.5 * a' * dt    # half-kick

    Velocity-Verlet is time-reversible and (nearly) energy-conserving, which
    is exactly why total energy staying flat is a good honesty check on a run.
    Returns (new_positions, new_velocities, new_forces).
    """
    accel = forces / mass
    v_half = velocities + 0.5 * accel * dt
    new_positions = positions + v_half * dt
    new_forces = lj_forces(new_positions, epsilon, sigma)
    new_accel = new_forces / mass
    new_velocities = v_half + 0.5 * new_accel * dt
    return new_positions, new_velocities, new_forces


# --------------------------------------------------------------------------
# 5. The loop: run a full trajectory
# --------------------------------------------------------------------------
def run_md(positions, velocities, dt=0.004, steps=2000,
           mass=1.0, epsilon=1.0, sigma=1.0, record_every=1):
    """Run molecular dynamics and record the trajectory and energies.

    Returns a dict with:
        traj      : (T, N, d) recorded positions,
        potential : (T,) potential energy,
        kinetic   : (T,) kinetic energy,
        total     : (T,) total energy (should be ~flat),
        temp      : (T,) instantaneous temperature.
    """
    positions = np.asarray(positions, dtype=float).copy()
    velocities = np.asarray(velocities, dtype=float).copy()
    forces = lj_forces(positions, epsilon, sigma)

    traj, pot, kin, tot, temp = [], [], [], [], []
    for step in range(steps):
        if step % record_every == 0:
            u = lj_energy(positions, epsilon, sigma)
            k = kinetic_energy(velocities, mass)
            traj.append(positions.copy())
            pot.append(u)
            kin.append(k)
            tot.append(u + k)
            temp.append(temperature(velocities, mass))
        positions, velocities, forces = velocity_verlet_step(
            positions, velocities, forces, dt, mass, epsilon, sigma)

    return {
        "traj": np.array(traj),
        "potential": np.array(pot),
        "kinetic": np.array(kin),
        "total": np.array(tot),
        "temp": np.array(temp),
    }


# --------------------------------------------------------------------------
# 6. Setting up a system: a starting cluster and some heat
# --------------------------------------------------------------------------
def triangular_lattice(nx: int, ny: int, spacing: float = 2.0 ** (1.0 / 6.0)) -> np.ndarray:
    """A 2D triangular (hex-packed) cluster of atoms, centered at the origin.

    The default spacing is 2**(1/6), the Lennard-Jones minimum, so the cluster
    starts already relaxed and will not explode on the first step.
    """
    pts = []
    for iy in range(ny):
        for ix in range(nx):
            x = ix * spacing + (0.5 * spacing if iy % 2 else 0.0)
            y = iy * spacing * np.sqrt(3.0) / 2.0
            pts.append((x, y))
    pts = np.array(pts, dtype=float)
    pts -= pts.mean(axis=0)          # center at the origin
    return pts


def disk_cluster(radius: float = 4.0, spacing: float = 2.0 ** (1.0 / 6.0)) -> np.ndarray:
    """A roughly circular blob of atoms — a triangular lattice trimmed to a disk.

    A round cluster is a bit nicer to watch than a rectangle, and it has no
    sharp corners for atoms to evaporate from.
    """
    n = int(2 * radius / spacing) + 2
    lattice = triangular_lattice(2 * n, 2 * n, spacing)
    return lattice[np.linalg.norm(lattice, axis=1) <= radius]


def thermal_velocities(n: int, d: int = 2, temp: float = 0.2,
                       mass: float = 1.0, seed: int = 0) -> np.ndarray:
    """Random velocities at a target temperature, with zero net momentum.

    We draw Gaussian velocities, remove the overall drift (so the cluster does
    not sail off the screen), then rescale so the temperature is exactly `temp`.
    """
    if n < 2:
        raise ValueError(
            "thermal_velocities needs at least 2 atoms: removing the net "
            "momentum of a single atom leaves it exactly at rest, so no finite "
            "temperature can be assigned (the rescale would divide by zero)."
        )
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, np.sqrt(temp / mass), size=(n, d))
    v -= v.mean(axis=0)                                   # zero total momentum
    current = 2.0 * (0.5 * mass * np.sum(v ** 2)) / (d * n)
    v *= np.sqrt(temp / current)                          # exact target temp
    return v
