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
#T_END   = 604800 * 5
DT      = 1

R2_INIT_BASE = np.array([5.0 * AU_TO_M, 0.0])
R1_INIT_X    = 0.0

xlsx_path = Path("BBH_SIM/InputParams/BHMergerSimulationParameters.xlsx")


def load_parameters(xlsx_path: Path) -> pd.DataFrame:
    return pd.read_excel(xlsx_path)


def build_simulation(row) -> BBHSimulation:
    #impact_m = float(row["Impact Parameter (m)"])
    impact_m = 1.496e13
    r1_init  = np.array([R1_INIT_X, impact_m])
    r2_init  = R2_INIT_BASE.copy()

    v1_init = np.array([
        2997924.58,
        float(row["BH1 Initial Y Velocity (m/s)"]),
    ])
    v2_init = np.array([
        -0,
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
        t_end    = float(row["Simulation Duration (s)"]),
        dt       = float(row["DT (s)"]),
        impact_m = impact_m,
    )


def run_single(run_id: int, save_plot: str = None, zoom_au: float = 0.5) -> None:
    """
    Run and visualize a single simulation by its ID.
    Parameters
    ----------
    run_id     : ID column value from the Excel sheet to run.
    save_plot  : Optional file path to save the plot (e.g. 'orbit_1.png').
                 If None, the plot is shown interactively.
    downsample : Plot every Nth point to keep the figure responsive.
                 Set to 1 to plot all points (slow for long runs).
    zoom_au    : Half-width of closest approach zoom window in AU (default: 0.5).
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
    print(f"r1_array_2d shape: {sim.r1_array_2d.shape}")
    print(f"r1 start: {sim.r1_array_2d[0]}")
    print(f"r1 end:   {sim.r1_array_2d[-1]}")
    print(f"r2 start: {sim.r2_array_2d[0]}")
    print(f"r2 end:   {sim.r2_array_2d[-1]}")
    print(f"[run_single] Done. merger={sim.merger_occurred}, "
          f"nearest_approach={sim.separation_distance / AU_TO_M:.4f} AU")

    r1 = sim.r1_array_2d / AU_TO_M
    r2 = sim.r2_array_2d / AU_TO_M

    plot_orbits_2d_rich(r1, r2, sim, run_id=run_id, save_path=save_plot, zoom_au=zoom_au)

    if save_plot:
        print(f"[run_single] Plot saved to {save_plot}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run and visualize a single BBH simulation.")
    parser.add_argument("run_id",   type=int,   help="Simulation ID to run")
    parser.add_argument("--zoom",   type=float, default=0.5)
    parser.add_argument("--save",   type=str,   default=None)
    args = parser.parse_args()

    run_single(args.run_id, save_plot=args.save, zoom_au=args.zoom)