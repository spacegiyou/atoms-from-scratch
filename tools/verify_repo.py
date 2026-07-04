#!/usr/bin/env python3
"""
verify_repo.py  --  one-shot health check for the atoms-from-scratch repo.

Run it from the repo root (the folder that contains README.md, afs/, notebooks/):

    python verify_repo.py

or point it at a folder:

    python verify_repo.py /path/to/atoms-from-scratch

It checks four things and prints a PASS/FAIL summary:
  1. Structure   -- required files/folders exist.
  2. Clean       -- no stray duplicates / caches / nested zip at the root.
  3. Fixes       -- notebooks 03/04/05 contain the corrected code and have outputs.
  4. Physics     -- the afs test suite passes (force == -grad(E); energy conserved).

Exit code is 0 only if every check passes.
"""
import sys, os, json, subprocess

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
ok_all = True
def line(status, msg):
    global ok_all
    mark = "PASS" if status else "FAIL"
    if not status:
        ok_all = False
    print(f"  [{mark}] {msg}")

def read_ipynb(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def cell_sources(nb):
    return ["".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"]

def has_outputs(nb):
    return any(c.get("cell_type") == "code" and c.get("outputs") for c in nb.get("cells", []))

print(f"\nChecking repo at: {ROOT}\n")

# ---------------------------------------------------------------- 1. structure
print("1) Structure")
required = [
    "README.md", "LICENSE", "requirements.txt",
    "afs/md.py", "afs/__init__.py",
    "tests/test_md.py",
    "notebooks/01_molecular_dynamics_from_scratch.ipynb",
    "notebooks/02_why_hand_written_potentials_break.ipynb",
    "notebooks/03_neural_network_potential_from_scratch.ipynb",
    "notebooks/04_why_symmetry_is_the_architecture.ipynb",
    "notebooks/05_message_passing_graph_neural_nets.ipynb",
    "notebooks/06_foundation_models_zero_shot.ipynb",
    "notebooks/07_when_your_simulation_is_lying.ipynb",
    "notebooks/08_capstone_real_property_end_to_end.ipynb",
    "assets/lj_cluster.gif",
]
for rel in required:
    line(os.path.exists(os.path.join(ROOT, rel)), f"exists: {rel}")

# ---------------------------------------------------------------- 2. clean root
print("\n2) Clean (no clutter at the root)")
root_entries = set(os.listdir(ROOT))
# stray duplicates that must NOT sit at the repo root
for stray in ["md.py", "test_md.py", "lj_cluster.gif",
              "01_molecular_dynamics_from_scratch.ipynb"]:
    line(stray not in root_entries, f"no root duplicate: {stray}")
# no zip archives anywhere in the tree
zips = [os.path.join(dp, fn) for dp, _, fns in os.walk(ROOT)
        for fn in fns if fn.endswith(".zip")]
line(len(zips) == 0, f"no .zip inside repo ({len(zips) or 'none'} found)")
# no python / pytest caches anywhere
caches = [os.path.join(dp, d) for dp, dns, _ in os.walk(ROOT)
          for d in dns if d in ("__pycache__", ".pytest_cache", ".ipynb_checkpoints")]
line(len(caches) == 0, f"no __pycache__/.pytest_cache ({len(caches) or 'none'} found)")
if caches:
    for c in caches:
        print(f"         -> delete: {os.path.relpath(c, ROOT)}")

# ---------------------------------------------------------------- 3. fixes
print("\n3) Fixes present in notebooks 03 / 04 / 05")
nbdir = os.path.join(ROOT, "notebooks")
try:
    nb3 = read_ipynb(os.path.join(nbdir, "03_neural_network_potential_from_scratch.ipynb"))
    src3 = "\n".join(cell_sources(nb3))
    line("torch.sort(d).values" in src3,
         "NB03: torch feature path is sorted (graph-preserving)")
    line(has_outputs(nb3), "NB03: has embedded outputs (executed)")
except Exception as e:
    line(False, f"NB03: could not read ({e})")

try:
    nb4 = read_ipynb(os.path.join(nbdir, "04_why_symmetry_is_the_architecture.ipynb"))
    src4 = "\n".join(cell_sources(nb4))
    line("Same sorted distances:" not in src4,
         "NB04: the false 'Same sorted distances' demo is gone")
    line("Permutation-invariant:" in src4,
         "NB04: correct permutation-invariance demo present")
    line(has_outputs(nb4), "NB04: has embedded outputs (executed)")
except Exception as e:
    line(False, f"NB04: could not read ({e})")

try:
    nb5 = read_ipynb(os.path.join(nbdir, "05_message_passing_graph_neural_nets.ipynb"))
    src5 = "\n".join(cell_sources(nb5))
    line("edge_vec = pos_t[dst_t] - pos_t[src_t]" in src5,
         "NB05: edge distances recomputed in torch (autograd fix)")
    line("_rbf_torch" in src5, "NB05: differentiable torch RBF present")
    line("grad is not None" in src5,
         "NB05: non-None gradient sanity check present")
    line(has_outputs(nb5), "NB05: has embedded outputs (executed)")
except Exception as e:
    line(False, f"NB05: could not read ({e})")

# ---------------------------------------------------------------- 4. physics
print("\n4) Physics (afs test suite)")
try:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    tail = (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else ""
    line(r.returncode == 0, f"pytest tests/ -> {tail or 'no output'}")
except FileNotFoundError:
    line(False, "pytest not installed (pip install pytest)")
except Exception as e:
    line(False, f"pytest could not run ({e})")

# ---------------------------------------------------------------- summary
print("\n" + "=" * 52)
print("RESULT:", "ALL CHECKS PASSED ✅" if ok_all else "SOME CHECKS FAILED ❌ (see above)")
print("=" * 52 + "\n")
sys.exit(0 if ok_all else 1)
