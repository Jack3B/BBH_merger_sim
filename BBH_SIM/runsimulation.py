import numpy as np
import pandas as pd
from pathlib import Path
from .simulation import BBHSimulation
from .datastorage import data_storage
from .dynamics import compute_positions

# ── Simulation time config (adjust as needed) ─────────────────────────────────
T_START = 0.0        # seconds
T_END   = 3.15576e10 # seconds (1000 yr)
DT      = 1e4        # seconds
r2_init = np.array([10],[0]) #adjust as need be with preliminary tests
r1_init_pre_param = np.array([0],[0]) #y gets adjusted by impact param

#inital pos for bh1 and bh2 then calc adjust for impact param for bh1

#xlsx_path = r"InputParams/BHMergerSimulationParameters.xlsx" #111,100 sims
xlsx_path = r"InputParams/BHMergerSimulationParameters_constvel.xlsx" #1111 sims lmao

def load_parameters(xlsx_path: str | Path) -> pd.DataFrame:
    return pd.read_excel(xlsx_path)

def build_simulation(row: pd.Series) -> BBHSimulation:
    
    #make compute_positions in dynamics to get r1 and r2 x and y components.
    
    return BBHSimulation(
        m1      = float(row["BH1 Mass (M_sol)"]),
        m2      = float(row["BH2 Mass (M_sol)"]),
        
        r1_init = r1_init,
        r2_init = r2_init,
        
        v1_init = np.array([
        float(row["BH1 Initial X Velocity (km/s)"]),
        float(row["BH1 Initial Y Velocity (km/s)"]),
    ])
        v2_init = np.array([
        float(row["BH2 Initial X Velocity (km/s)"]),
        float(row["BH2 Initial Y Velocity (km/s)"]),
    ])
        
        t_start = T_START,
        t_end   = T_END,
        dt      = DT,
       
def run_all (
    
    #ADD THIS LATER