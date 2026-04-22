import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch


def plot_orbits_2d(r1_array, r2_array, show=True, save_path=None):
    """
    Plot the 2D orbits of the binary black holes.
    """
    fig, ax = plt.subplots()
    ax.plot(
        r1_array[:, 0],
        r1_array[:, 1],
        "o-",
        color="blue",
        markersize=5,
        label="Black Hole 1",
    )
    ax.plot(
        r2_array[:, 0],
        r2_array[:, 1],
        "^-",
        color="orange",
        markersize=5,
        label="Black Hole 2",
    )
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def animate_trajectories_2d(r1_array, r2_array, save_path=None):
    """
    Animate the 2D trajectories of the binary black holes.
    """
    fig, ax = plt.subplots()
    (line1,) = ax.plot([], [], "o-", color="blue", markersize=5, label="Black Hole 1")
    (line2,) = ax.plot([], [], "^-", color="orange", markersize=5, label="Black Hole 2")
    ax.set_xlim(
        min(np.min(r1_array[:, 0]), np.min(r2_array[:, 0])),
        max(np.max(r1_array[:, 0]), np.max(r2_array[:, 0])),
    )
    ax.set_ylim(
        min(np.min(r1_array[:, 1]), np.min(r2_array[:, 1])),
        max(np.max(r1_array[:, 1]), np.max(r2_array[:, 1])),
    )
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return (
            line1,
            line2,
        )

    def update(frame):
        line1.set_data(r1_array[:frame, 0], r1_array[:frame, 1])
        line2.set_data(r2_array[:frame, 0], r2_array[:frame, 1])
        return (
            line1,
            line2,
        )

    ani = FuncAnimation(
        fig, update, frames=len(r1_array), init_func=init, blit=True, repeat=False
    )

    if save_path:
        ani.save(save_path, writer="imagemagick", fps=30)
    else:
        plt.show()

    return ani


def plot_orbits_2d_rich(r1_au, r2_au, sim, run_id, save_path=None, zoom_au=0.5):
    AU_TO_M         = 1.495978707e11
    SOLARMASS_TO_KG = 1.989e30

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0d0d0d")

    ax_full = fig.add_subplot(2, 2, 1)
    ax_zoom = fig.add_subplot(2, 2, 3)
    ax_info = fig.add_subplot(2, 2, (2, 4))

    def _style(ax):
        ax.set_facecolor("#0d0d0d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.tick_params(colors="#aaa")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.title.set_color("white")
        ax.grid(True, color="#222", lw=0.5)

    _style(ax_full)
    _style(ax_zoom)
    ax_info.axis("off")

    sep_au = sim.separation_distance / AU_TO_M
    sep_t  = sim.separation_time

    r_sch1_au = sim.r_sch1 / AU_TO_M
    r_sch2_au = sim.r_sch2 / AU_TO_M

    def add_motion_arrows(ax, traj, color, n_arrows=5):
        """Add evenly spaced arrows along a trajectory indicating direction of motion."""
        n = len(traj)
        if n < 2:
            return
        indices = np.linspace(0, n - 2, n_arrows, dtype=int)
        for i in indices:
            dx = traj[i+1, 0] - traj[i, 0]
            dy = traj[i+1, 1] - traj[i, 1]
            ax.annotate(
                "",
                xy=(traj[i+1, 0], traj[i+1, 1]),
                xytext=(traj[i, 0], traj[i, 1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.2,
                ),
                zorder=7,
            )

    # ── Full trajectory ───────────────────────────────────────────────────────
    ax_full.plot(r1_au[:, 0], r1_au[:, 1], color="#4fa3e0", lw=0.8, alpha=0.85, label="BH1")
    ax_full.plot(r2_au[:, 0], r2_au[:, 1], color="#e07b4f", lw=0.8, alpha=0.85, label="BH2", linestyle="--")
    add_motion_arrows(ax_full, r1_au, "#4fa3e0", n_arrows=5)
    add_motion_arrows(ax_full, r2_au, "#e07b4f", n_arrows=5)

    # Start/end markers
    ax_full.plot(*r1_au[0],  "o", color="#4fa3e0", ms=8,  zorder=5)
    ax_full.plot(*r2_au[0],  "o", color="#e07b4f", ms=8,  zorder=5)
    ax_full.plot(*r1_au[-1], "x", color="#4fa3e0", ms=10, mew=2, zorder=5)
    ax_full.plot(*r2_au[-1], "x", color="#e07b4f", ms=10, mew=2, zorder=5)

    # Schwarzschild radius circles at final positions (full plot)
    from matplotlib.patches import Circle
    for pos, r_sch, color in [
        (r1_au[-1], r_sch1_au, "#4fa3e0"),
        (r2_au[-1], r_sch2_au, "#e07b4f"),
    ]:
        ax_full.add_patch(Circle(pos, r_sch, fill=False,
                                 edgecolor=color, lw=1.0,
                                 linestyle="-", alpha=0.7, zorder=6))

    ax_full.set_xlabel("X (AU)")
    ax_full.set_ylabel("Y (AU)")
    ax_full.set_title(f"Run {run_id} — Full Trajectory")
    ax_full.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="white", fontsize=9)

    # ── Closest approach zoom ─────────────────────────────────────────────────
    dists  = np.linalg.norm(r1_au - r2_au, axis=1)
    ca_idx = np.argmin(dists)
    if dists[ca_idx] < 1e-10:
        ca_idx = len(r1_au) // 2
    ca_x = (r1_au[ca_idx, 0] + r2_au[ca_idx, 0]) / 2
    ca_y = (r1_au[ca_idx, 1] + r2_au[ca_idx, 1]) / 2

    ax_zoom.plot(r1_au[:, 0], r1_au[:, 1], color="#4fa3e0", lw=1.0, alpha=0.9, label="BH1")
    ax_zoom.plot(r2_au[:, 0], r2_au[:, 1], color="#e07b4f", lw=1.0, alpha=0.9, label="BH2", linestyle="--")
    add_motion_arrows(ax_zoom, r1_au, "#4fa3e0", n_arrows=3)
    add_motion_arrows(ax_zoom, r2_au, "#e07b4f", n_arrows=3)

    ax_zoom.plot(r1_au[ca_idx, 0], r1_au[ca_idx, 1], "o", color="#4fa3e0", ms=7, zorder=6)
    ax_zoom.plot(r2_au[ca_idx, 0], r2_au[ca_idx, 1], "o", color="#e07b4f", ms=7, zorder=6)

    # Schwarzschild radius circles at closest approach positions (zoom plot)
    for pos, r_sch, color in [
        (r1_au[ca_idx], r_sch1_au, "#4fa3e0"),
        (r2_au[ca_idx], r_sch2_au, "#e07b4f"),
    ]:
        ax_zoom.add_patch(Circle(pos, r_sch, fill=False,
                                 edgecolor=color, lw=1.2,
                                 linestyle="-", alpha=0.9, zorder=6))

    # Line connecting BHs at closest approach
    ax_zoom.plot(
        [r1_au[ca_idx, 0], r2_au[ca_idx, 0]],
        [r1_au[ca_idx, 1], r2_au[ca_idx, 1]],
        "--", color="white", lw=0.8, alpha=0.6,
    )
    ax_zoom.annotate(
        f"{sep_au:.6f} AU",
        xy=(ca_x, ca_y),
        xytext=(ca_x + zoom_au * 0.15, ca_y + zoom_au * 0.15),
        color="white", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
    )

    ax_zoom.set_xlim(ca_x - zoom_au, ca_x + zoom_au)
    ax_zoom.set_ylim(ca_y - zoom_au, ca_y + zoom_au)
    ax_zoom.set_xlabel("X (AU)")
    ax_zoom.set_ylabel("Y (AU)")
    ax_zoom.set_title(f"Closest Approach  (±{zoom_au} AU)")
    ax_zoom.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="white", fontsize=8)
    ax_zoom.set_aspect("equal", adjustable="datalim")

    # Rectangle on full plot showing zoom region
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (ca_x - zoom_au, ca_y - zoom_au),
        2 * zoom_au, 2 * zoom_au,
        linewidth=1, edgecolor="yellow", facecolor="none",
        linestyle="--", zorder=6,
    )
    ax_full.add_patch(rect)

    # ── Info panel ────────────────────────────────────────────────────────────
    m1_sol        = sim.m1 / SOLARMASS_TO_KG
    m2_sol        = sim.m2 / SOLARMASS_TO_KG
    impact_au     = sim.impact_m / AU_TO_M
    v1_init_kms   = np.linalg.norm(sim.v1_init) / 1e3
    v2_init_kms   = np.linalg.norm(sim.v2_init) / 1e3
    v1_fin_kms    = np.linalg.norm(sim.v1) / 1e3
    v2_fin_kms    = np.linalg.norm(sim.v2) / 1e3
    duration_days = (len(sim.r1_array) * sim.dt) / 86400
    remain_au     = (sim.distance_needed_for_merger / AU_TO_M
                     if sim.distance_needed_for_merger else float("nan"))

    lines = [
        ("Run ID",                   f"{run_id}"),
        ("",                         ""),
        ("── Bodies ──",             ""),
        ("BH1 mass",                 f"{m1_sol:.2f} M☉"),
        ("BH2 mass",                 f"{m2_sol:.2f} M☉"),
        ("BH1 Schwarzschild r",      f"{sim.r_sch1/1e3:.1f} km"),
        ("BH2 Schwarzschild r",      f"{sim.r_sch2/1e3:.1f} km"),
        ("",                         ""),
        ("── Initial conditions ──", ""),
        ("Impact parameter",         f"{impact_au:.6f} AU"),
        ("BH1 v₀",                   f"{v1_init_kms:.2f} km/s"),
        ("BH2 v₀",                   f"{v2_init_kms:.2f} km/s"),
        ("",                         ""),
        ("── Outcome ──",            ""),
        ("Merged",                   "YES ✓" if sim.merger_occurred else "NO ✗"),
        ("Nearest approach",         f"{sep_au:.6f} AU"),
        ("Nearest approach time",    f"{sep_t/86400:.3f} days"),
        ("Remaining to merge",       f"{remain_au:.6f} AU"),
        ("BH1 v_final",              f"{v1_fin_kms:.2f} km/s"),
        ("BH2 v_final",              f"{v2_fin_kms:.2f} km/s"),
        ("BH1 deflection",           f"{sim.r1_deflection_angle:.4f}°"),
        ("BH2 deflection",           f"{sim.r2_deflection_angle:.4f}°"),
        ("",                         ""),
        ("── Simulation ──",         ""),
        ("Duration",                 f"{duration_days:.3f} days"),
        ("dt",                       f"{sim.dt} s"),
        ("Steps",                    f"{len(sim.r1_array):,}"),
        ("Zoom window",              f"±{zoom_au} AU"),
    ]

    y = 0.97
    for label, value in lines:
        if label.startswith("──"):
            ax_info.text(0.02, y, label, color="#888", fontsize=8.5,
                         transform=ax_info.transAxes, va="top", style="italic")
        elif label != "":
            ax_info.text(0.02, y, label, color="#aaa", fontsize=9,
                         transform=ax_info.transAxes, va="top")
            ax_info.text(0.55, y, value, color="white", fontsize=9,
                         transform=ax_info.transAxes, va="top", weight="bold")
        y -= 0.045

    fig.suptitle(
        f"BBH Flyby Simulation — Run {run_id}",
        color="white", fontsize=13, weight="bold",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[plot] saved to {save_path}")
    else:
        plt.show()


def plot_from_file(file_path, plot_type="orbits", **kwargs):
    """
    Plot the simulation data from a file.

    Parameters:
    - file_path (str): Path to the simulation data file.
    - plot_type (str): Type of plot to generate ('orbits' or 'waveform').
    - **kwargs: Additional keyword arguments to pass to the respective plotting function.
    """
    data = np.loadtxt(file_path)
    t_array = data[:, 0]
    r1_array = data[:, 1:4]
    r2_array = data[:, 4:7]

    if plot_type == "orbits":
        plot_orbits_3d(r1_array, r2_array, **kwargs)
    elif plot_type == "waveform":
        h_plus = data[:, 7]
        h_cross = data[:, 8]
        plot_waveform(t_array, h_plus, h_cross, **kwargs)
    else:
        raise ValueError(
            f"Invalid plot type: {plot_type}. Supported types are 'orbits' and 'waveform'."
        )
