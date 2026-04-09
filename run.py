import time
from BBH_SIM.runsimulation import run_all
from pathlib import Path

start = time.time()

run_all(
    xlsx_path   = Path("BBH_SIM/InputParams/BHMergerSimulationParameters_constvel_testrun.xlsx"),
    output_path = Path("BBH_SIM/Results/bbh_results.h5"),
    start_id    = 1,
)

elapsed = time.time() - start
hours   = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = elapsed % 60
print(f"\nTotal time: {hours}h {minutes}m {seconds:.2f}s")