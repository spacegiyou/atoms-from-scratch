"""Build notebooks/04_why_symmetry_is_the_architecture.ipynb  (CORRECTED)

Fix vs original: Section 1 claimed to build "two configurations with identical
sorted pairwise distances but physically distinct", but the code actually built a
triangle (distances 2,2,2) and a collinear chain (distances 2,2,4) -- which have
DIFFERENT sorted distances, so the printed `Same sorted distances:` was False and
the narrative contradicted the output. (It is also subtly wrong for the LJ target,
whose energy is a function of the distance multiset, so sorted distances are not
representationally insufficient for LJ specifically.)

The corrected Section 1 demonstrates the *real, verifiable* limitations of sorted
distances -- permutation invariance (the good property), non-smoothness (kinks when
the sort order swaps, which hurts force learning), no locality/cutoff, and no
angular information for genuine many-body chemistry -- which is the honest motivation
for symmetry functions. Sections 2-5 are unchanged in substance.
"""

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

nb = new_notebook()
cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

md(
    "# 04 -- Why symmetry *is* the architecture\n"
    "\n"
    "**The physics symmetries are the inductive bias. Build them in and you need far less data.**\n"
    "\n"
    "In Notebook 03 we used sorted pairwise distances as features. That was a quick fix with three "
    "real weaknesses, which we make concrete below. The principled cure is to build the symmetries "
    "and the locality of physics directly into the representation.\n"
    "\n"
    "The three symmetries every atomic potential must have:\n"
    "\n"
    "1. **Translation invariance** -- shifting all atoms does not change the energy.\n"
    "2. **Rotation invariance** -- rotating the whole system does not change the energy.\n"
    "3. **Permutation invariance** -- swapping two identical atoms does not change the energy.\n"
    "\n"
    "Behler and Parrinello (2007) introduced **symmetry functions** that satisfy all three, are "
    "smooth, are local (they use a cutoff), and -- crucially for real chemistry -- encode "
    "**angular** information that pairwise distances alone cannot. In this notebook we implement "
    "them from scratch and show they improve data efficiency over raw distances."
)

code(
    "import numpy as np\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "from afs.md import (\n"
    "    lj_energy, lj_forces, triangular_lattice,\n"
    "    disk_cluster, thermal_velocities, run_md,\n"
    ")\n"
    "\n"
    "torch.manual_seed(0)\n"
    "np.random.seed(0)"
)

# ── Section 1 (CORRECTED): the real weaknesses of sorted distances ─────────
md(
    "## 1. What sorted distances get right -- and where they fail\n"
    "\n"
    "Sorted pairwise distances (Notebook 03) are genuinely translation-, rotation-, and "
    "permutation-invariant. Let us confirm the permutation part, then expose the three problems "
    "that motivate a better representation."
)

code(
    "def sorted_distances(pos):\n"
    "    n = len(pos)\n"
    "    dists = [np.linalg.norm(pos[i] - pos[j])\n"
    "             for i in range(n) for j in range(i + 1, n)]\n"
    "    return np.sort(dists)\n"
    "\n"
    "rng = np.random.default_rng(0)\n"
    "pos = triangular_lattice(3, 3) + 0.1 * rng.standard_normal((9, 2))\n"
    "\n"
    "# (Good property) Permutation invariance: relabel the atoms -> identical feature vector\n"
    "perm = rng.permutation(9)\n"
    "print('Permutation-invariant:', np.allclose(sorted_distances(pos), sorted_distances(pos[perm])))"
)

md(
    "**Problem 1 -- non-smoothness.** Sorting introduces kinks: as an atom moves, two distances "
    "can cross and swap places in the sorted vector. At that point the feature has a corner, so its "
    "derivative jumps -- and forces are derivatives. Kinky features make force learning harder. "
    "Watch a few components of the sorted-distance vector as we slide one atom along a line."
)

code(
    "base = triangular_lattice(3, 3)\n"
    "shifts = np.linspace(-1.2, 1.2, 300)\n"
    "comps = np.array([sorted_distances(base + np.array([[s, 0.0]] + [[0, 0]] * 8)) for s in shifts])\n"
    "\n"
    "plt.figure(figsize=(7, 3))\n"
    "for c in [0, 5, 10, 20]:\n"
    "    plt.plot(shifts, comps[:, c], lw=1.5, label=f'sorted dist #{c}')\n"
    "plt.xlabel('displacement of atom 0 along x'); plt.ylabel('distance value')\n"
    "plt.title('Sorted-distance features have kinks (where the sort order swaps)')\n"
    "plt.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
    "print('Each corner is a point where two distances cross and swap slots -- a discontinuous')\n"
    "print('derivative. Smooth symmetry functions (below) avoid this.')"
)

md(
    "**Problem 2 -- no locality.** The sorted-distance vector has length $N(N-1)/2$: it grows as "
    "$N^2$ and mixes every pair, near and far. Real potentials are *local* -- an atom only feels "
    "its neighbours within a cutoff -- which keeps cost $O(N)$ and makes the model transferable "
    "across system sizes.\n"
    "\n"
    "**Problem 3 -- no angular information.** For a genuine many-body potential (real chemistry, "
    "DFT), the energy depends on bond *angles*, not just distances. Two local environments can "
    "share a distance distribution yet differ in their angles. Pairwise distances cannot see that; "
    "the angular symmetry functions below can.\n"
    "\n"
    "> (For the Lennard-Jones toy target, whose energy is literally a sum over pair distances, "
    "distances are in principle sufficient -- so here the payoff of symmetry functions shows up as "
    "smoothness and data efficiency. On real many-body data, angular terms become *necessary*.)"
)

# ── Section 2: Behler-Parrinello symmetry functions ──────────────────────
md(
    "## 2. Behler-Parrinello symmetry functions\n"
    "\n"
    "The Behler-Parrinello (BP) approach represents the local environment of atom $i$ by a "
    "fingerprint built from two kinds of functions:\n"
    "\n"
    "**Radial** (how many neighbours at each distance):\n"
    "$$G^{rad}_{i,\\mu} = \\sum_{j \\neq i} e^{-\\eta (r_{ij} - r_s)^2} \\, f_c(r_{ij})$$\n"
    "\n"
    "**Angular** (which angles are present):\n"
    "$$G^{ang}_{i,\\mu} = 2^{1-\\zeta} \\sum_{j,k \\neq i} (1 + \\lambda \\cos\\theta_{ijk})^\\zeta "
    "\\, e^{-\\eta (r_{ij}^2 + r_{ik}^2 + r_{jk}^2)} \\, f_c(r_{ij}) f_c(r_{ik}) f_c(r_{jk})$$\n"
    "\n"
    "with a smooth cutoff $f_c(r) = 0.5[\\cos(\\pi r / r_c) + 1]$ for $r < r_c$, else 0.\n"
    "\n"
    "These are translation-, rotation-, and permutation-invariant, **smooth** (the cutoff and "
    "Gaussians have continuous derivatives), **local** (the cutoff), and carry **angular** "
    "information -- fixing all three problems above."
)

code(
    "def cutoff(r, r_cut=3.5):\n"
    "    \"\"\"Cosine cutoff: smooth to zero at r_cut.\"\"\"\n"
    "    return np.where(r < r_cut, 0.5 * (np.cos(np.pi * r / r_cut) + 1.0), 0.0)\n"
    "\n"
    "def g2_radial(pos, r_cut=3.5, eta_list=None, rs_list=None):\n"
    "    \"\"\"Radial symmetry functions for all atoms. Returns (N, n_eta*n_rs).\"\"\"\n"
    "    if eta_list is None:\n"
    "        eta_list = [0.5, 1.0, 2.0, 4.0]\n"
    "    if rs_list is None:\n"
    "        rs_list = [0.0, 0.5, 1.0, 1.5, 2.0]\n"
    "    n = len(pos)\n"
    "    feats = []\n"
    "    for i in range(n):\n"
    "        row = []\n"
    "        for eta in eta_list:\n"
    "            for rs in rs_list:\n"
    "                G = 0.0\n"
    "                for j in range(n):\n"
    "                    if j == i:\n"
    "                        continue\n"
    "                    rij = np.linalg.norm(pos[i] - pos[j])\n"
    "                    G += np.exp(-eta * (rij - rs) ** 2) * cutoff(rij, r_cut)\n"
    "                row.append(G)\n"
    "        feats.append(row)\n"
    "    return np.array(feats)\n"
    "\n"
    "def g4_angular(pos, r_cut=3.5, eta=0.5, zeta_list=None, lam_list=None):\n"
    "    \"\"\"Angular symmetry functions (simplified G4) for all atoms. Returns (N, n_zeta*n_lam).\"\"\"\n"
    "    if zeta_list is None:\n"
    "        zeta_list = [1.0, 4.0]\n"
    "    if lam_list is None:\n"
    "        lam_list = [-1.0, 1.0]\n"
    "    n = len(pos)\n"
    "    feats = []\n"
    "    for i in range(n):\n"
    "        row = []\n"
    "        for zeta in zeta_list:\n"
    "            for lam in lam_list:\n"
    "                G = 0.0\n"
    "                for j in range(n):\n"
    "                    if j == i:\n"
    "                        continue\n"
    "                    for k in range(n):\n"
    "                        if k == i or k <= j:\n"
    "                            continue\n"
    "                        rij_vec = pos[j] - pos[i]\n"
    "                        rik_vec = pos[k] - pos[i]\n"
    "                        rij = np.linalg.norm(rij_vec)\n"
    "                        rik = np.linalg.norm(rik_vec)\n"
    "                        rjk = np.linalg.norm(pos[j] - pos[k])\n"
    "                        cos_theta = np.dot(rij_vec, rik_vec) / (rij * rik + 1e-12)\n"
    "                        G += (2 ** (1 - zeta)\n"
    "                              * (1 + lam * cos_theta) ** zeta\n"
    "                              * np.exp(-eta * (rij ** 2 + rik ** 2 + rjk ** 2))\n"
    "                              * cutoff(rij, r_cut) * cutoff(rik, r_cut) * cutoff(rjk, r_cut))\n"
    "                row.append(G)\n"
    "        feats.append(row)\n"
    "    return np.array(feats)\n"
    "\n"
    "def bp_features(pos):\n"
    "    \"\"\"Concatenate radial + angular features per atom, then sum over atoms.\"\"\"\n"
    "    per_atom = np.concatenate([g2_radial(pos), g4_angular(pos)], axis=1)\n"
    "    return per_atom.sum(axis=0)\n"
    "\n"
    "# BP features are invariant to translation and rotation\n"
    "p = triangular_lattice(3, 3) + 0.1 * rng.standard_normal((9, 2))\n"
    "feat = bp_features(p)\n"
    "feat_shifted = bp_features(p + np.array([10.0, 5.0]))\n"
    "theta = np.pi / 3\n"
    "R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])\n"
    "feat_rotated = bp_features((R @ p.T).T)\n"
    "print(f'BP feature length: {len(feat)}')\n"
    "print('Translation-invariant:', np.allclose(feat, feat_shifted, atol=1e-6))\n"
    "print('Rotation-invariant:   ', np.allclose(feat, feat_rotated, atol=1e-5))\n"
    "print('Permutation-invariant:', np.allclose(feat, bp_features(p[rng.permutation(9)]), atol=1e-6))"
)

# ── Section 3: Train with BP features ────────────────────────────────────
md(
    "## 3. Train a BP-featured network and compare to raw distances\n"
    "\n"
    "Now we train two networks on the same dataset -- one on sorted pairwise distances "
    "(Notebook 03), one on BP symmetry functions -- and sweep the training-set size to measure "
    "data efficiency."
)

code(
    "def make_dataset_bp(n_samples=800, noise=0.3):\n"
    "    base = triangular_lattice(3, 3)\n"
    "    rng = np.random.default_rng(42)\n"
    "    X_dist, X_bp, Y = [], [], []\n"
    "    while len(Y) < n_samples:\n"
    "        pos = base + noise * rng.standard_normal(base.shape)\n"
    "        E = lj_energy(pos)\n"
    "        if abs(E) > 50:\n"
    "            continue\n"
    "        X_dist.append(sorted_distances(pos))\n"
    "        X_bp.append(bp_features(pos))\n"
    "        Y.append(E)\n"
    "    return np.array(X_dist), np.array(X_bp), np.array(Y)\n"
    "\n"
    "print('Generating dataset (this takes ~30 s)...')\n"
    "X_dist, X_bp, Y = make_dataset_bp(n_samples=800)\n"
    "print(f'Dataset: {len(Y)} configs, dist features: {X_dist.shape[1]}, BP features: {X_bp.shape[1]}')"
)

code(
    "def train_net(X_tr, y_tr, X_va, y_va, hidden=64, epochs=300, lr=3e-3):\n"
    "    model = nn.Sequential(\n"
    "        nn.Linear(X_tr.shape[1], hidden), nn.SiLU(),\n"
    "        nn.Linear(hidden, hidden), nn.SiLU(),\n"
    "        nn.Linear(hidden, 1),\n"
    "    )\n"
    "    opt = torch.optim.Adam(model.parameters(), lr=lr)\n"
    "    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)\n"
    "    Xtr = torch.tensor(X_tr, dtype=torch.float32); ytr = torch.tensor(y_tr, dtype=torch.float32)\n"
    "    Xva = torch.tensor(X_va, dtype=torch.float32); yva = torch.tensor(y_va, dtype=torch.float32)\n"
    "    for _ in range(epochs):\n"
    "        model.train(); opt.zero_grad()\n"
    "        loss = nn.functional.mse_loss(model(Xtr).squeeze(), ytr)\n"
    "        loss.backward(); opt.step(); sched.step()\n"
    "    model.eval()\n"
    "    with torch.no_grad():\n"
    "        return nn.functional.mse_loss(model(Xva).squeeze(), yva).item() ** 0.5\n"
    "\n"
    "train_sizes = [50, 100, 200, 400, 600]\n"
    "n_val = 150\n"
    "val_i = np.arange(len(Y) - n_val, len(Y))\n"
    "rmse_dist, rmse_bp = [], []\n"
    "for n_tr in train_sizes:\n"
    "    tr_i = np.arange(n_tr)\n"
    "    rd = train_net(X_dist[tr_i], Y[tr_i], X_dist[val_i], Y[val_i])\n"
    "    rb = train_net(X_bp[tr_i], Y[tr_i], X_bp[val_i], Y[val_i])\n"
    "    rmse_dist.append(rd); rmse_bp.append(rb)\n"
    "    print(f'n_train={n_tr:4d}  dist RMSE={rd:.4f}  BP RMSE={rb:.4f}')\n"
    "\n"
    "plt.figure(figsize=(6, 4))\n"
    "plt.loglog(train_sizes, rmse_dist, 'o-', label='sorted distances (Nb 03)')\n"
    "plt.loglog(train_sizes, rmse_bp, 's-', label='BP symmetry functions')\n"
    "plt.xlabel('training set size'); plt.ylabel('val RMSE (energy)')\n"
    "plt.title('Data efficiency: BP vs raw distances')\n"
    "plt.legend(); plt.tight_layout(); plt.show()"
)

# ── Section 4: the intuition ──────────────────────────────────────────────
md(
    "## 4. What symmetry buys you\n"
    "\n"
    "By encoding what we know about the physics (symmetries, locality, smoothness) in the "
    "representation, the network only has to learn what the physics does *not* hand us -- the "
    "shape of the energy surface. This is the heart of modern ML for science:\n"
    "\n"
    "> The best ML models for physics are not the most flexible ones. They are the ones that "
    "encode the right physical constraints.\n"
    "\n"
    "**Radial functions** are Gaussian probes: the feature counts how many atoms sit in each "
    "distance band -- a density fingerprint. **Angular functions** add a bond-angle fingerprint "
    "that breaks the degeneracy between different bonding geometries."
)

code(
    "r_vals = np.linspace(0.1, 3.5, 400)\n"
    "eta_list = [0.5, 1.0, 2.0, 4.0]\n"
    "rs_list = [0.0, 0.5, 1.0, 1.5, 2.0]\n"
    "fig, axes = plt.subplots(2, 2, figsize=(10, 7))\n"
    "for ax, eta in zip(axes.flat, eta_list):\n"
    "    for rs in rs_list:\n"
    "        G = np.exp(-eta * (r_vals - rs) ** 2) * cutoff(r_vals)\n"
    "        ax.plot(r_vals, G, lw=1.5, label=f'rs={rs}')\n"
    "    ax.set_title(f'Radial G2 (eta={eta})')\n"
    "    ax.set_xlabel('r'); ax.set_ylabel('G2'); ax.legend(fontsize=7)\n"
    "plt.suptitle('Radial symmetry functions: smooth Gaussian probes at different distances', fontsize=12)\n"
    "plt.tight_layout(); plt.show()"
)

# ── Section 5: wrap-up ────────────────────────────────────────────────────
md(
    "## 5. What comes next\n"
    "\n"
    "BP symmetry functions solve the symmetry problem elegantly but have two remaining issues:\n"
    "\n"
    "1. **Hand-designed features.** The radial and angular functions are chosen by hand and may "
    "   miss important structure.\n"
    "2. **Not end-to-end trainable.** The features are fixed before training; the network cannot "
    "   adapt the representation to the data.\n"
    "\n"
    "The fix is **graph neural networks** (GNNs): instead of fixed symmetry functions, the network "
    "learns its own atom representations by passing messages along the bonds of the molecular "
    "graph. Each atom aggregates information from its neighbours -- exactly what the BP sum does, "
    "but now the aggregation is itself learned. That is Notebook 05.\n"
    "\n"
    "### Exercises\n"
    "\n"
    "1. **More features.** Add more $(\\eta, r_s)$ pairs to the radial functions. When does adding "
    "   more stop helping?\n"
    "\n"
    "2. **Per-atom energies.** In BP, $E = \\sum_i E_i(\\mathbf{G}_i)$. Implement this: one network "
    "   mapping the per-atom feature $\\mathbf{G}_i$ to a per-atom energy $E_i$, then sum. This is "
    "   more physical and scales to large systems.\n"
    "\n"
    "3. **Forces via autograd.** Implement `bp_features_torch` in PyTorch keeping the graph, and "
    "   verify `torch.autograd` gives forces consistent with finite differences (as in Notebook 03)."
)

nb.cells = cells
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "notebooks", "04_why_symmetry_is_the_architecture.ipynb")
with open(OUT, "w") as f:
    nbf.write(nb, f)
print(f"wrote {OUT}")
