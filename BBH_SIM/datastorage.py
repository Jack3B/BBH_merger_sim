from __future__ import annotations
import h5py
import numpy as np
from pathlib import Path

COLUMNS = [
    ("run_id",                          np.int64),
    ("bh1_mass_msol",                   np.float64),
    ("bh2_mass_msol",                   np.float64),
    ("impact_parameter_au",             np.float64),
    ("bh1_schwarzschild_radius_km",     np.float64),
    ("bh2_schwarzschild_radius_km",     np.float64),
    ("simulation_duration_yr",          np.float64),

    # BH1
    ("bh1_v_init_x_kms",               np.float64),
    ("bh1_v_init_y_kms",               np.float64),
    ("bh1_v_init_total_kms",           np.float64),
    ("bh1_v_init_unit_x",              np.float64),
    ("bh1_v_init_unit_y",              np.float64),
    ("bh1_v_final_x_kms",              np.float64),
    ("bh1_v_final_y_kms",              np.float64),
    ("bh1_v_final_total_kms",          np.float64),
    ("bh1_v_final_unit_x",             np.float64),
    ("bh1_v_final_unit_y",             np.float64),
    ("bh1_deflection_angle_deg",       np.float64),
    
    # BH2
    ("bh2_v_init_x_kms",               np.float64),
    ("bh2_v_init_y_kms",               np.float64),
    ("bh2_v_init_total_kms",           np.float64),
    ("bh2_v_init_unit_x",              np.float64),
    ("bh2_v_init_unit_y",              np.float64),
    ("bh2_v_final_x_kms",              np.float64),
    ("bh2_v_final_y_kms",              np.float64),
    ("bh2_v_final_total_kms",          np.float64),
    ("bh2_v_final_unit_x",             np.float64),
    ("bh2_v_final_unit_y",             np.float64),
    ("bh2_deflection_angle_deg",       np.float64),
    
    # global outcomes
    ("merged",                          np.int8),       # 0 or 1
    ("nearest_approach_dist_au",        np.float64),
    ("nearest_approach_time_s",         np.float64),
    ("remaining_dist_for_merger_au",    np.float64),
]

DTYPE = np.dtype([(name, typ) for name, typ in COLUMNS])


class data_storage:

    DATASET = "results"

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self._ensure_file()

    def append(self, **fields) -> int:

        row = self._build_row(fields)
        with h5py.File(self.filepath, "a") as f:
            ds = f[self.DATASET]
            idx = ds.shape[0]
            ds.resize(idx + 1, axis=0)
            ds[idx] = row
        return idx

    def read_all(self) -> np.ndarray:
        with h5py.File(self.filepath, "r") as f:
            return f[self.DATASET][:]

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.read_all())

    def __len__(self) -> int:
        with h5py.File(self.filepath, "r") as f:
            return f[self.DATASET].shape[0]


    def _ensure_file(self):
        with h5py.File(self.filepath, "a") as f:
            if self.DATASET not in f:
                f.create_dataset(
                    self.DATASET,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=DTYPE,
                    chunks=(256,),
                    compression="gzip",
                    compression_opts=4,
                )
                f[self.DATASET].attrs["columns"] = [c for c, _ in COLUMNS]

    def _build_row(self, fields: dict) -> np.ndarray:
        missing = {name for name, _ in COLUMNS} - fields.keys()
        if missing:
            raise ValueError(f"Missing fields for HDF5 row: {sorted(missing)}")

        row = np.zeros(1, dtype=DTYPE)
        for name, _ in COLUMNS:
            row[name] = fields[name]
        return row