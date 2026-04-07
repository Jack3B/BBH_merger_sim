import numpy as np
from .dynamics import compute_acceleration
from .dynamics import compute_schwarzschild_radii
from .dynamics import compute_merger_event_test
from .datastorage import data_storage

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
        r_sch1=None,
        r_sch2=None,
    ):
        self.m1 = m1
        self.m2 = m2
        self.r1 = r1_init
        self.r2 = r2_init
        self.v1 = v1_init
        self.v2 = v2_init
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
        
        self._r1_init = r1_init.copy()
        self._r2_init = r2_init.copy()
        self._v1_init = v1_init.copy()
        self._v2_init = v2_init.copy()

        # ── Populated after run() ─────────────────────────────────────────────
        self._r_sch1 = None
        self._r_sch2 = None
        self._merger_step = None #merger time
        
    def run(self):
        
        r_sch1 = compute_schwarzschild_radii(self.m1)
        r_sch2 = compute_schwarzschild_radii(self.m2)
        
        for _t in self.t_array:
            r = self.r2 - self.r1
            v = self.v2 - self.v1

            spins = (self.spin1, self.spin2) if self.spin else None

            a1 = compute_acceleration(
                r, v, self.m1, self.m2, self.pn_order, self.radiation, spins
            )
            a2 = -compute_acceleration(
                r, v, self.m1, self.m2, self.pn_order, self.radiation, spins
            )
            
            merger = compute_merger_event_test(r, r_sch1, r_sch2)
            
            if merger:
                self.merger_occurred = True
                self._merger_step = step
                break
            
            # Update velocities
            self.v1 += a1 * self.dt
            self.v2 += a2 * self.dt

            # Update positions
            self.r1 += self.v1 * self.dt
            self.r2 += self.v2 * self.dt

            # Store positions
            self.r1_array.append(self.r1.copy())
            self.r2_array.append(self.r2.copy())
          

            if self.r1.size == 3:  # Check if the input positions are 3D
                self.r1_array_2d.append(self.r1[:2].copy())
                self.r2_array_2d.append(self.r2[:2].copy())
            else:  # If the input positions are 2D, directly store them
                self.r1_array_2d.append(self.r1.copy())
                self.r2_array_2d.append(self.r2.copy())

        self.r1_array = np.array(self.r1_array)
        self.r2_array = np.array(self.r2_array)
        self.r1_array_2d = np.array(self.r1_array_2d)
        self.r2_array_2d = np.array(self.r2_array_2d)

     def save_results(self, store, run_id: int) -> int:
        """
        Compute all derived quantities and append one row to a BBHResultsHDF5
        store.  Returns the row index written.

        Parameters
        ----------
        store   : BBHResultsHDF5 instance
        run_id  : integer identifier for this run
        """
        if self._r_sch1 is None:
            raise RuntimeError("Call run() before save_results().")

        # ── BH1 velocity stats ─────────────────────────────────────────────
        (bh1_vx_i, bh1_vy_i, bh1_vtot_i, bh1_ux_i, bh1_uy_i,
         bh1_vx_f, bh1_vy_f, bh1_vtot_f, bh1_ux_f, bh1_uy_f,
         bh1_defl) = self._velocity_stats(self._v1_init, self.v1)

        # ── BH2 velocity stats ─────────────────────────────────────────────
        (bh2_vx_i, bh2_vy_i, bh2_vtot_i, bh2_ux_i, bh2_uy_i,
         bh2_vx_f, bh2_vy_f, bh2_vtot_f, bh2_ux_f, bh2_uy_f,
         bh2_defl) = self._velocity_stats(self._v2_init, self.v2)

        nearest_dist_au, nearest_time_s = self._compute_nearest_approach()

        idx = store.append(
            run_id                      = int(run_id),
            bh1_mass_msol               = float(self.m1),
            bh2_mass_msol               = float(self.m2),
            impact_parameter_au         = self._compute_impact_parameter_au(),
            bh1_schwarzschild_radius_km = float(self._r_sch1) / M_PER_KM,
            bh2_schwarzschild_radius_km = float(self._r_sch2) / M_PER_KM,
            # BH1 velocities
            bh1_v_init_x_kms            = bh1_vx_i,
            bh1_v_init_y_kms            = bh1_vy_i,
            bh1_v_init_total_kms        = bh1_vtot_i,
            bh1_v_init_unit_x           = bh1_ux_i,
            bh1_v_init_unit_y           = bh1_uy_i,
            bh1_v_final_x_kms           = bh1_vx_f,
            bh1_v_final_y_kms           = bh1_vy_f,
            bh1_v_final_total_kms       = bh1_vtot_f,
            bh1_v_final_unit_x          = bh1_ux_f,
            bh1_v_final_unit_y          = bh1_uy_f,
            bh1_deflection_angle_deg    = bh1_defl,
            # BH2 velocities
            bh2_v_init_x_kms            = bh2_vx_i,
            bh2_v_init_y_kms            = bh2_vy_i,
            bh2_v_init_total_kms        = bh2_vtot_i,
            bh2_v_init_unit_x           = bh2_ux_i,
            bh2_v_init_unit_y           = bh2_uy_i,
            bh2_v_final_x_kms           = bh2_vx_f,
            bh2_v_final_y_kms           = bh2_vy_f,
            bh2_v_final_total_kms       = bh2_vtot_f,
            bh2_v_final_unit_x          = bh2_ux_f,
            bh2_v_final_unit_y          = bh2_uy_f,
            bh2_deflection_angle_deg    = bh2_defl,
            # Outcome
            merged                      = int(self.merger_occurred),
            nearest_approach_dist_au    = nearest_dist_au,
            nearest_approach_time_s     = nearest_time_s,
            remaining_dist_for_merger_au= self._compute_remaining_dist_au(),
            simulation_duration_yr      = self._compute_simulation_duration_yr(),
        )
        return idx            
