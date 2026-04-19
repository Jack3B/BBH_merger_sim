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


def plot_orbits_3d(
    r1_array, r2_array, fig=None, ax=None, show=True, save_path=None, **kwargs
):
    """
    Plot the 3D orbits of the binary black holes.

    Parameters:
    - r1_array (numpy.ndarray): Array of position vectors of the first black hole.
    - r2_array (numpy.ndarray): Array of position vectors of the second black hole.
    - fig (matplotlib.figure.Figure): Figure object to use for plotting (default: None).
    - ax (matplotlib.axes.Axes): Axes object to use for plotting (default: None).
    - show (bool): Whether to display the plot (default: True).
    - save_path (str): Path to save the plot (default: None).
    - **kwargs: Additional keyword arguments to pass to the plotting function.
    """
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        r1_array[:, 0], r1_array[:, 1], r1_array[:, 2], label="Black Hole 1", **kwargs
    )
    ax.plot(
        r2_array[:, 0], r2_array[:, 1], r2_array[:, 2], label="Black Hole 2", **kwargs
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def animate_trajectories_3d(r1_array, r2_array, save_path=None):
    """
    Animate the 3D trajectories of the binary black holes.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    (line1,) = ax.plot(
        [], [], [], "o-", color="blue", markersize=5, label="Black Hole 1"
    )
    (line2,) = ax.plot(
        [], [], [], "^-", color="orange", markersize=5, label="Black Hole 2"
    )

    ax.set_xlim(
        min(np.min(r1_array[:, 0]), np.min(r2_array[:, 0])),
        max(np.max(r1_array[:, 0]), np.max(r2_array[:, 0])),
    )
    ax.set_ylim(
        min(np.min(r1_array[:, 1]), np.min(r2_array[:, 1])),
        max(np.max(r1_array[:, 1]), np.max(r2_array[:, 1])),
    )
    ax.set_zlim(
        min(np.min(r1_array[:, 2]), np.min(r2_array[:, 2])),
        max(np.max(r1_array[:, 2]), np.max(r2_array[:, 2])),
    )

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.legend()

    def init():
        line1.set_data([], [])
        line1.set_3d_properties([])
        line2.set_data([], [])
        line2.set_3d_properties([])
        return line1, line2

    def update(frame):
        line1.set_data(r1_array[:frame, 0], r1_array[:frame, 1])
        line1.set_3d_properties(r1_array[:frame, 2])
        line2.set_data(r2_array[:frame, 0], r2_array[:frame, 1])
        line2.set_3d_properties(r2_array[:frame, 2])
        return line1, line2

    ani = FuncAnimation(
        fig, update, frames=len(r1_array), init_func=init, blit=True, repeat=False
    )

    if save_path:
        ani.save(save_path, writer="imagemagick", fps=30)
    else:
        plt.show()

    return ani


def plot_waveform(
    t_array, h_plus, h_cross, fig=None, ax=None, show=True, save_path=None, **kwargs
):
    """
    Plot the gravitational waveform.

    Parameters:
    - t_array (numpy.ndarray): Array of time values.
    - h_plus (numpy.ndarray): Plus polarization of the gravitational waveform.
    - h_cross (numpy.ndarray): Cross polarization of the gravitational waveform.
    - fig (matplotlib.figure.Figure): Figure object to use for plotting (default: None).
    - ax (matplotlib.axes.Axes): Axes object to use for plotting (default: None).
    - show (bool): Whether to display the plot (default: True).
    - save_path (str): Path to save the plot (default: None).
    - **kwargs: Additional keyword arguments to pass to the plotting function.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    ax.plot(t_array, h_plus, label="Plus Polarization", **kwargs)
    ax.plot(t_array, h_cross, label="Cross Polarization", **kwargs)
    ax.set_xlabel("Time")
    ax.set_ylabel("Strain")
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

def plot_orbits_2d_rich(r1_au, r2_au, sim, run_id, save_path=None):
    """
    r1_au, r2_au : already downsampled, in AU
    sim          : BBHSimulation instance (post-run)
    """
    AU_TO_M      = 1.495978707e11
    SOLARMASS_TO_KG = 1.989e30

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#0d0d0d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.tick_params(colors="#aaa")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.title.set_color("white")

    # ── LEFT: full trajectory ─────────────────────────────────────────────────
    ax = axes[0]

    ax.plot(r1_au[:, 0], r1_au[:, 1], color="#4fa3e0", lw=0.8, alpha=0.85, label="BH1")
    ax.plot(r2_au[:, 0], r2_au[:, 1], color="#e07b4f", lw=0.8, alpha=0.85, label="BH2")

    # Start markers
    ax.plot(*r1_au[0],  "o", color="#4fa3e0", ms=8, zorder=5)
    ax.plot(*r2_au[0],  "o", color="#e07b4f", ms=8, zorder=5)

    # End markers
    ax.plot(*r1_au[-1], "x", color="#4fa3e0", ms=10, mew=2, zorder=5)
    ax.plot(*r2_au[-1], "x", color="#e07b4f", ms=10, mew=2, zorder=5)

    # Closest approach marker
    sep_au = sim.separation_distance / AU_TO_M
    sep_t  = sim.separation_time
    ax.annotate(
        f"Closest approach\n{sep_au:.4f} AU",
        xy=(r1_au[0, 0], r1_au[0, 1]),   # approximate — just flags the BH1 start
        xytext=(r1_au[0, 0] + 0.3, r1_au[0, 1] + 0.3),
        color="white", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
    )

    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_title(f"Run {run_id} — Full Trajectory")
    ax.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="white", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#222", lw=0.5)

    # ── RIGHT: info panel ─────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")

    m1_sol = sim.m1 / SOLARMASS_TO_KG
    m2_sol = sim.m2 / SOLARMASS_TO_KG
    impact_au = sim.impact_m / AU_TO_M

    v1_init_kms = np.linalg.norm(sim.v1_init) / 1e3
    v2_init_kms = np.linalg.norm(sim.v2_init) / 1e3
    v1_fin_kms  = np.linalg.norm(sim.v1) / 1e3
    v2_fin_kms  = np.linalg.norm(sim.v2) / 1e3

    duration_days = (len(sim.r1_array) * sim.dt) / 86400
    remain_au = sim.distance_needed_for_merger / AU_TO_M if sim.distance_needed_for_merger else float("nan")

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
    ]

    y = 0.97
    for label, value in lines:
        if label.startswith("──"):
            ax2.text(0.02, y, label, color="#888", fontsize=8.5,
                     transform=ax2.transAxes, va="top", style="italic")
        elif label == "":
            pass
        else:
            ax2.text(0.02, y, label,  color="#aaa",   fontsize=9,
                     transform=ax2.transAxes, va="top")
            ax2.text(0.55, y, value,  color="white",  fontsize=9,
                     transform=ax2.transAxes, va="top", weight="bold")
        y -= 0.045

    merged_color = "#50e87a" if sim.merger_occurred else "#e85050"
    fig.suptitle(
        f"BBH Flyby Simulation — Run {run_id}",
        color="white", fontsize=13, weight="bold", y=1.01
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
