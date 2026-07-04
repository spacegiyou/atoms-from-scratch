"""Generate the figures and hero animation used in the README.

Run from the repo root:  python assets/make_assets.py
Produces:
    assets/lj_potential.png       the iconic Lennard-Jones curve
    assets/energy_conservation.png potential / kinetic / total vs time
    assets/lj_cluster.gif          the hero: a cluster of atoms doing MD
    assets/_preview.png            a single frame, for quick visual checks
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from afs.md import lj_energy, run_md, disk_cluster, thermal_velocities

HERE = os.path.dirname(os.path.abspath(__file__))
BG = "#0d1117"      # GitHub dark background
FG = "#c9d1d9"
ACCENT = "#58a6ff"


def fig_lj_potential():
    r = np.linspace(0.9, 3.0, 400)
    v = 4.0 * (r ** -12 - r ** -6)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.axhline(0, color=FG, lw=0.6, alpha=0.4)
    ax.plot(r, v, color=ACCENT, lw=2.5)
    rmin = 2 ** (1 / 6)
    ax.scatter([rmin], [-1], color="#f778ba", zorder=5, s=40)
    ax.annotate("minimum at r = 2^(1/6)\nenergy = -1",
                xy=(rmin, -1), xytext=(1.5, -0.6), color=FG, fontsize=10,
                arrowprops=dict(color=FG, arrowstyle="->", alpha=0.7))
    ax.set_ylim(-1.4, 2.0)
    ax.set_xlabel("distance between two atoms  r", color=FG)
    ax.set_ylabel("potential energy  V(r)", color=FG)
    ax.set_title("The Lennard-Jones potential", color=FG, fontsize=13)
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_color(FG); s.set_alpha(0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "lj_potential.png"), facecolor=BG)
    plt.close(fig)
    print("wrote lj_potential.png")


def run_hero():
    atoms = disk_cluster(radius=4.6)
    vel = thermal_velocities(len(atoms), d=2, temp=0.40, seed=2)
    out = run_md(atoms, vel, dt=0.005, steps=1320, record_every=12)
    print(f"hero run: {len(atoms)} atoms, {len(out['traj'])} frames")
    return out


def fig_energy_conservation(out):
    t = np.arange(len(out["total"]))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.plot(t, out["potential"], color="#f778ba", lw=1.5, label="potential")
    ax.plot(t, out["kinetic"], color="#3fb950", lw=1.5, label="kinetic")
    ax.plot(t, out["total"], color=ACCENT, lw=2.5, label="total (flat = honest)")
    ax.set_xlabel("recorded step", color=FG)
    ax.set_ylabel("energy", color=FG)
    ax.set_title("Energy is conserved: the simulation is honest", color=FG, fontsize=13)
    ax.tick_params(colors=FG)
    leg = ax.legend(facecolor=BG, edgecolor="none", labelcolor=FG, fontsize=9)
    for s in ax.spines.values():
        s.set_color(FG); s.set_alpha(0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "energy_conservation.png"), facecolor=BG)
    plt.close(fig)
    print("wrote energy_conservation.png")


def make_gif(out, preview_only=False):
    traj = out["traj"]
    # color atoms by instantaneous speed (frame-to-frame displacement)
    speeds = np.linalg.norm(np.diff(traj, axis=0, prepend=traj[:1]), axis=2)
    vmax = np.percentile(speeds, 98)

    lim = 6.5
    fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.axis("off")

    scat = ax.scatter(traj[0][:, 0], traj[0][:, 1], s=320,
                      c=speeds[0], cmap="plasma", vmin=0, vmax=vmax,
                      edgecolors="white", linewidths=0.4)
    txt = ax.text(-lim + 0.3, lim - 0.7, "molecular dynamics from scratch",
                  color=FG, fontsize=11, alpha=0.8)

    def update(frame):
        scat.set_offsets(traj[frame])
        scat.set_array(speeds[frame])
        return scat, txt

    if preview_only:
        update(len(traj) // 2)
        fig.savefig(os.path.join(HERE, "_preview.png"), facecolor=BG)
        plt.close(fig)
        print("wrote _preview.png")
        return

    anim = FuncAnimation(fig, update, frames=len(traj), interval=45, blit=False)
    anim.save(os.path.join(HERE, "lj_cluster.gif"),
              writer=PillowWriter(fps=20), dpi=92)
    plt.close(fig)
    print("wrote lj_cluster.gif")


if __name__ == "__main__":
    fig_lj_potential()
    out = run_hero()
    fig_energy_conservation(out)
    if "--preview" in sys.argv:
        make_gif(out, preview_only=True)
    else:
        make_gif(out)
    print("done")
