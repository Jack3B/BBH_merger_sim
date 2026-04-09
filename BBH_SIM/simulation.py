import numpy as np
from .dynamics import compute_acceleration
from .dynamics import compute_schwarzschild_radii
from .dynamics import compute_merger_event_test
from .dynamics import compute_unit_vector
from .dynamics import compute_distance
from .dynamics import compute_remaining_distance_for_merger
from .dynamics import compute_deflection_angle

class BBHSimulation:
    def __init__(
        self,
        m1,
        m2,
        r1_init,
        r2_init,
        v1_init,
        v2_init,
        t_start,
        t_end,
        dt,
        pn_order=0,
        radiation=False,
        spin=False,
        spin1=None,
        spin2=None,
    ):
        self.m1 = m1
        self.m2 = m2
        self.r1 = r1_init.copy()
        self.r2 = r2_init.copy()
        self.v1 = v1_init.copy()
        self.v2 = v2_init.copy()

        # Snapshots for deflection angle calculation post-run
        self.r1_init = r1_init.copy()
        self.r2_init = r2_init.copy()
        self.v1_init = v1_init.copy()
        self.v2_init = v2_init.copy()

        # Initial unit vectors
        self.r1_unit_vector_x_init, self.r1_unit_vector_y_init = compute_unit_vector(r1_init)
        self.r2_unit_vector_x_init, self.r2_unit_vector_y_init = compute_unit_vector(r2_init)
        self.v1_unit_vector_x_init, self.v1_unit_vector_y_init = compute_unit_vector(v1_init)
        self.v2_unit_vector_x_init, self.v2_unit_vector_y_init = compute_unit_vector(v2_init)

        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.pn_order = pn_order
        self.radiation = radiation
        self.spin = spin
        self.spin1 = spin1
        self.spin2 = spin2

        self.t_array = np.arange(t_start, t_end + dt, dt)
        self.r1_array = []
        self.r2_array = []
        self.r1_array_2d = []
        self.r2_array_2d = []
        self.merger_occurred = False

        # Populated during/after run()
        self.r_sch1 = None
        self.r_sch2 = None
        self.separation_distance = compute_distance(self.r1, self.r2)
        self.separation_time = t_start
        self.distance_needed_for_merger = None

        # Final unit vectors and deflection angles — set after run()
        self.r1_unit_vector_x_fin = None
        self.r1_unit_vector_y_fin = None
        self.r2_unit_vector_x_fin = None
        self.r2_unit_vector_y_fin = None
        self.v1_unit_vector_x_fin = None
        self.v1_unit_vector_y_fin = None
        self.v2_unit_vector_x_fin = None
        self.v2_unit_vector_y_fin = None
        self.r1_deflection_angle = None
        self.r2_deflection_angle = None
        
def run(self):
        self.r_sch1, self.r_sch2 = compute_schwarzschild_radii(self.m1, self.m2)

        for step, _t in enumerate(self.t_array):
            r = self.r2 - self.r1
            v = self.v2 - self.v1
            spins = (self.spin1, self.spin2) if self.spin else None

            a1 = compute_acceleration(
                r, v, self.m1, self.m2, self.pn_order, self.radiation, spins
            )
            a2 = -compute_acceleration(
                r, v, self.m1, self.m2, self.pn_order, self.radiation, spins
            )

            merger = compute_merger_event_test(self.r1, self.r2, self.r_sch1, self.r_sch2)

            if merger:
                self.merger_occurred = True
                break

            self.v1 += a1 * self.dt
            self.v2 += a2 * self.dt
            self.r1 += self.v1 * self.dt
            self.r2 += self.v2 * self.dt

            # Track nearest approach — update as long as distance is decreasing
            current_dist = compute_distance(self.r1, self.r2)
            if current_dist < self.separation_distance:
                self.separation_distance = current_dist
                self.separation_time = _t

            self.r1_array.append(self.r1.copy())
            self.r2_array.append(self.r2.copy())

            if self.r1.size == 3:
                self.r1_array_2d.append(self.r1[:2].copy())
                self.r2_array_2d.append(self.r2[:2].copy())
            else:
                self.r1_array_2d.append(self.r1.copy())
                self.r2_array_2d.append(self.r2.copy())

        self.r1_array    = np.array(self.r1_array)
        self.r2_array    = np.array(self.r2_array)
        self.r1_array_2d = np.array(self.r1_array_2d)
        self.r2_array_2d = np.array(self.r2_array_2d)

        # Final unit vectors
        self.r1_unit_vector_x_fin, self.r1_unit_vector_y_fin = compute_unit_vector(self.r1)
        self.r2_unit_vector_x_fin, self.r2_unit_vector_y_fin = compute_unit_vector(self.r2)
        self.v1_unit_vector_x_fin, self.v1_unit_vector_y_fin = compute_unit_vector(self.v1)
        self.v2_unit_vector_x_fin, self.v2_unit_vector_y_fin = compute_unit_vector(self.v2)

        # Deflection angles
        self.r1_deflection_angle = compute_deflection_angle(
            (self.r1_unit_vector_x_init, self.r1_unit_vector_y_init),
            (self.r1_unit_vector_x_fin,  self.r1_unit_vector_y_fin),
        )
        self.r2_deflection_angle = compute_deflection_angle(
            (self.r2_unit_vector_x_init, self.r2_unit_vector_y_init),
            (self.r2_unit_vector_x_fin,  self.r2_unit_vector_y_fin),
        )

        # Remaining distance needed for merger at closest approach
        self.distance_needed_for_merger = compute_remaining_distance_for_merger(
            self.separation_distance, self.r_sch1, self.r_sch2
        )

    def save_results(self, store, run_id: int) -> int:
        """Append one row to a data_storage store directly from sim attributes."""
        if self.r_sch1 is None:
            raise RuntimeError("Call run() before save_results().")

        return store.append(
            run_id                          = run_id,
            bh1_mass_msol                   = self.m1,
            bh2_mass_msol                   = self.m2,
            impact_parameter_au             = float(np.linalg.norm(self.r1_init) / AU_TO_M),
            bh1_schwarzschild_radius_km     = self.r_sch1 / KM_TO_M,
            bh2_schwarzschild_radius_km     = self.r_sch2 / KM_TO_M,
            bh1_v_init_x_kms               = self.v1_init[0] / KM_TO_M,
            bh1_v_init_y_kms               = self.v1_init[1] / KM_TO_M,
            bh1_v_init_total_kms           = float(np.linalg.norm(self.v1_init)) / KM_TO_M,
            bh1_v_init_unit_x              = self.v1_unit_vector_x_init,
            bh1_v_init_unit_y              = self.v1_unit_vector_y_init,
            bh1_v_final_x_kms              = self.v1[0] / KM_TO_M,
            bh1_v_final_y_kms              = self.v1[1] / KM_TO_M,
            bh1_v_final_total_kms          = float(np.linalg.norm(self.v1)) / KM_TO_M,
            bh1_v_final_unit_x             = self.v1_unit_vector_x_fin,
            bh1_v_final_unit_y             = self.v1_unit_vector_y_fin,
            bh1_deflection_angle_deg       = self.r1_deflection_angle,
            bh2_v_init_x_kms               = self.v2_init[0] / KM_TO_M,
            bh2_v_init_y_kms               = self.v2_init[1] / KM_TO_M,
            bh2_v_init_total_kms           = float(np.linalg.norm(self.v2_init)) / KM_TO_M,
            bh2_v_init_unit_x              = self.v2_unit_vector_x_init,
            bh2_v_init_unit_y              = self.v2_unit_vector_y_init,
            bh2_v_final_x_kms              = self.v2[0] / KM_TO_M,
            bh2_v_final_y_kms              = self.v2[1] / KM_TO_M,
            bh2_v_final_total_kms          = float(np.linalg.norm(self.v2)) / KM_TO_M,
            bh2_v_final_unit_x             = self.v2_unit_vector_x_fin,
            bh2_v_final_unit_y             = self.v2_unit_vector_y_fin,
            bh2_deflection_angle_deg       = self.r2_deflection_angle,
            merged                         = int(self.merger_occurred),
            nearest_approach_dist_au       = self.separation_distance / AU_TO_M,
            nearest_approach_time_s        = self.separation_time,
            remaining_dist_for_merger_au   = self.distance_needed_for_merger / AU_TO_M,
            simulation_duration_yr         = (len(self.r1_array) * self.dt) / S_PER_YR,
        )

    def save_data(self, filename):
        data = np.column_stack(
            (self.r1_array, self.r2_array, self.r1_array_2d, self.r2_array_2d)
        )
        np.savetxt(filename, data, header=f"merger_occurred={int(self.merger_occurred)}")

    def load_data(self, filename):
        with open(filename, "r") as f:
            header = f.readline()
        self.merger_occurred = bool(int(header.strip().split("=")[1]))
        data = np.loadtxt(filename)
        self.r1_array    = data[:, :3]
        self.r2_array    = data[:, 3:6]
        self.r1_array_2d = data[:, 6:8]
        self.r2_array_2d = data[:, 8:]
