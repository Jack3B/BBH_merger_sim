from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from BBH_SIM.simulation import BBHSimulation
from BBH_SIM.visualization import plot_orbits_2d
from BBH_SIM.visualization import plot_orbits_2d_rich


# ── Unit conversions ──────────────────────────────────────────────────────────
AU_TO_M = 1.495978707e11
KM_TO_M = 1e3

# ── Simulation time config ────────────────────────────────────────────────────
T_START = 0.0
T_END   = 2.592e+6 * 1.5
DT      = 10

R2_INIT_BASE = np.array([5.0 * AU_TO_M, 0.0])
R1_INIT_X    = 0.0

xlsx_path = Path("BBH_SIM/InputParams/BHMergerSimulationParameters.xlsx")


def load_parameters(xlsx_path: Path) -> pd.DataFrame:
    return pd.read_excel(xlsx_path)


def build_simulation(row) -> BBHSimulation:
    impact_m = float(row["Impact Parameter (m)"])
    r1_init  = np.array([R1_INIT_X, impact_m])
    r2_init  = R2_INIT_BASE.copy()

    v1_init = np.array([
        200000,
        float(row["BH1 Initial Y Velocity (m/s)"]),
    ])
    v2_init = np.array([
        float(row["BH2 Initial X Velocity (m/s)"]),
        float(row["BH2 Initial Y Velocity (m/s)"]),
    ])

    return BBHSimulation(
        m1       = float(row["BH1 Mass (kg)"]),
        m2       = float(row["BH2 Mass (kg)"]),
        r1_init  = r1_init,
        r2_init  = r2_init,
        v1_init  = v1_init,
        v2_init  = v2_init,
        t_start  = T_START,
        t_end    = T_END,
        dt       = DT,
        impact_m = impact_m,
    )


def run_single(run_id: int, save_plot: str = None, downsample: int = 100) -> None:
    """
    Run and visualize a single simulation by its ID.

    Parameters
    ----------
    run_id     : ID column value from the Excel sheet to run.
    save_plot  : Optional file path to save the plot (e.g. 'orbit_1.png').
                 If None, the plot is shown interactively.
    downsample : Plot every Nth point to keep the figure responsive.
                 Set to 1 to plot all points (slow for long runs).
    """
    params = load_parameters(xlsx_path)
    row = params[params["ID"] == run_id]

    if row.empty:
        print(f"[error] No row found with ID={run_id}")
        return

    row = row.iloc[0]
    print(f"[run_single] Running ID={run_id}...")

    sim = build_simulation(row)
    sim.run()

    print(f"[run_single] Done. merger={sim.merger_occurred}, "
          f"nearest_approach={sim.separation_distance / AU_TO_M:.4f} AU")

    # Downsample so matplotlib doesn't choke on millions of points
    r1 = sim.r1_array_2d[::downsample] / AU_TO_M
    r2 = sim.r2_array_2d[::downsample] / AU_TO_M

    plot_orbits_2d_rich(r1, r2, sim, run_id=run_id, save_path=save_plot)

    if save_plot:
        print(f"[run_single] Plot saved to {save_plot}")
        print(f"BH1 x range: {r1_au[:, 0].min():.4f} to {r1_au[:, 0].max():.4f} AU")
        print(f"BH1 y range: {r1_au[:, 1].min():.4f} to {r1_au[:, 1].max():.4f} AU")
        print(f"BH2 x range: {r2_au[:, 0].min():.4f} to {r2_au[:, 0].max():.4f} AU")
        print(f"BH2 y range: {r2_au[:, 1].min():.4f} to {r2_au[:, 1].max():.4f} AU")
        print(f"BH1 first 5 positions:\n{r1_au[:5]}")
        print(f"BH2 first 5 positions:\n{r2_au[:5]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run and visualize a single BBH simulation.")
    parser.add_argument("run_id",               type=int,   help="Simulation ID to run")
    parser.add_argument("--save",               type=str,   default=None,
                        help="Path to save plot image (e.g. orbit_1.png)")
    parser.add_argument("--downsample",         type=int,   default=100,
                        help="Plot every Nth step (default: 100)")
    args = parser.parse_args()

    run_single(args.run_id, save_plot=args.save, downsample=args.downsample)