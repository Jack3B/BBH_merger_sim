#runsimulation.py
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from pathlib import Path
from BBH_SIM.simulation import BBHSimulation
from BBH_SIM.datastorage import data_storage
 
# ── Unit conversions ──────────────────────────────────────────────────────────
AU_TO_M = 1.495978707e11    # AU  → m
KM_TO_M = 1e3               # km/s → m/s
 
# ── Simulation time config ────────────────────────────────────────────────────
T_START = 0.0
#T_END   = 604800
#DT      = 1
 
# BH2 starts at x = 5 AU, y = 0
R2_INIT_BASE = np.array([5.0 * AU_TO_M, 0.0])
 
# BH1 starts at x = 0, y = impact_parameter (set per row)
R1_INIT_X = 0.0
 
xlsx_path = Path("InputParams/BHMergerSimulationParameters.xlsx")
#xlsx_path = Path("InputParams/BHMergerSimulationParameters_1it.xlsx")
 
OUTPUT_HDF5    = Path("BBH_SIM/results/bbh_results.h5")
#OUTPUT_HDF5   = Path("BBH_SIM/results/bbh_results_1it.h5")
 
 
def clear_outputs(*paths: Path) -> None:
    """Delete output files so the run starts completely fresh."""
    for p in paths:
        if p.exists():
            p.unlink()
            print(f"[setup] cleared {p}")
        else:
            print(f"[setup] {p} not found — nothing to clear")
 
 
def load_parameters(xlsx_path: Path) -> pd.DataFrame:
    return pd.read_excel(xlsx_path)
 
 
def build_simulation(row) -> BBHSimulation:
    impact_m = float(row["Impact Parameter (m)"])
    r1_init = np.array([R1_INIT_X, impact_m])
    r2_init = R2_INIT_BASE.copy()
 
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
 
 
def run_all(
    xlsx_path:   Path,
    output_path: Path,
    start_id:    int  = 1,
    clear:       bool = False,
) -> None:
    if clear:
        clear_outputs(output_path)

    params = load_parameters(xlsx_path)
    store  = data_storage(output_path)

    pending = params[params["ID"] >= start_id]
    if pending.empty:
        print(f"[run_all] No rows with ID >= {start_id}. Nothing to do.")
        return

    print(f"[run_all] Starting from ID {start_id} "
          f"({len(pending)} of {len(params)} runs remaining)")

    current_mass             = None
    consecutive_no_merge     = 0
    override_dt              = False

    for _, row in pending.iterrows():
        run_id   = int(row["ID"])
        row_mass = float(row["BH1 Mass (kg)"])

        # Reset when mass group changes
        if row_mass != current_mass:
            current_mass         = row_mass
            consecutive_no_merge = 0
            override_dt          = False

        # Build simulation, optionally overriding DT
        sim = build_simulation(row)
        if override_dt:
            sim.dt      = 10
            sim.t_array = np.arange(sim.t_start, sim.t_end + sim.dt, sim.dt)

        t_iter_start   = time.time()
        sim.run()
        t_iter_elapsed = max(0.0, time.time() - t_iter_start)

        sim.save_results(store, run_id=run_id)

        iter_hours,   rem          = divmod(t_iter_elapsed, 3600)
        iter_minutes, iter_seconds = divmod(rem, 60)
        dt_note = " [dt=10 override]" if override_dt else ""
        print(f"[{run_id}/{len(params)}] merged={sim.merger_occurred} | "
              f"iter time: {int(iter_hours)}h {int(iter_minutes)}m {iter_seconds:.2f}s{dt_note}")

        if sim.merger_occurred:
            consecutive_no_merge = 0
            override_dt          = False
        else:
            consecutive_no_merge += 1
            if consecutive_no_merge >= 3:
                override_dt = True
                print(f"[run_all] 3 consecutive non-mergers for mass {current_mass:.3e} "
                      f"— switching to dt=10 for remaining impact parameters")
 
 
if __name__ == "__main__":
    run_all(
        xlsx_path,
        output_path = OUTPUT_HDF5,
        start_id    = START_ID,
        clear       = CLEAR_ON_START,
    )
 