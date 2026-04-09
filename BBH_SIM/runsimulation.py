import numpy as np
import pandas as pd
from pathlib import Path
from .simulation import BBHSimulation
from .datastorage import data_storage
from .dynamics import compute_positions

# ── Simulation time config ────────────────────────────────────────────────────
T_START = 0.0
T_END   = 3.15576e10        # seconds (1000 yr)
DT      = 1e4               # seconds

# BH2 starts at x = -10 AU, y = 0
R2_INIT_BASE = np.array([-10.0 * AU_TO_M, 0.0])

# BH1 starts at x = 0, y = impact_parameter (set per row)
R1_INIT_X = 0.0

#xlsx_path = Path("InputParams/BHMergerSimulationParameters.xlsx") #111000
#xlsx_path = Path("InputParams/BHMergerSimulationParameters_constvel.xlsx") #1110
#xlsx_path = Path("InputParams/BHMergerSimulationParameters_constvel_testrun.xlsx") #9 iterations for testing
xlsx_path = Path("InputParams/BHMergerSimulationParameters_constvel_1it.xlsx") #1 iteration for testing lol

def load_parameters(xlsx_path: str | Path) -> pd.DataFrame:
    return pd.read_excel(xlsx_path)


def build_simulation(row: pd.Series) -> BBHSimulation:
    impact_m = float(row["Impact Parameter (AU)"]) * AU_TO_M

    r1_init = np.array([R1_INIT_X, impact_m])
    r2_init = R2_INIT_BASE.copy()

    v1_init = np.array([
        float(row["BH1 Initial X Velocity (km/s)"]) * KM_TO_M,
        float(row["BH1 Initial Y Velocity (km/s)"]) * KM_TO_M,
    ])
    v2_init = np.array([
        float(row["BH2 Initial X Velocity (km/s)"]) * KM_TO_M,
        float(row["BH2 Initial Y Velocity (km/s)"]) * KM_TO_M,
    ])

    return BBHSimulation(
        m1      = float(row["BH1 Mass (M_sol)"]),
        m2      = float(row["BH2 Mass (M_sol)"]),
        r1_init = r1_init,
        r2_init = r2_init,
        v1_init = v1_init,
        v2_init = v2_init,
        t_start = T_START,
        t_end   = T_END,
        dt      = DT,
    )


def run_all(xlsx_path: str | Path, output_path: str | Path, start_id: int = 1):
    """
    Run all simulations from the xlsx and append results to the HDF5 store.
    Set start_id = len(store) + 1 to resume an interrupted run.
    """
    params = load_parameters(xlsx_path)
    store  = data_storage(output_path)

    for _, row in params[params["ID"] >= start_id].iterrows():
        run_id = int(row["ID"])
        sim = build_simulation(row)
        sim.run()
        sim.save_results(store, run_id=run_id)
        print(f"[{run_id}/{len(params)}] merged={sim.merger_occurred}")


if __name__ == "__main__":
    run_all(xlsx_path, output_path=Path("results/bbh_results.h5"))