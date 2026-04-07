import numpy as np
from .dynamics import compute_acceleration
from .dynamics import compute_schwarzschild_radii
from .dynamics import compute_merger_event_test
from .dynamics import compute_unit_vector
from .dynamics import compute_distance

class BBHSimulation:
    def __init__(
        self,
        m1,
        m2,
        r1_init,
        r2_init,
        v1_init,
        v2_init,
        r1_fin,
        r2_fin,
        v1_fin,
        v2_fin,
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
        separation_distance=0,
        separation_time=0
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
        self._merger_step = None   # step index at which merger was detected
        self._r_sch1 = None
        self._r_sch2 = None
        self.separation_distance = compute_distance(r1, r2)
        self.separation_time = 0
        
    def run(self):
        
        r_sch1, r_sch2 = compute_schwarzschild_radii(self.m1, self.m2)
        
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
            
            merger = compute_merger_event_test(self.r1, self.r2, r_sch1, r_sch2)
            
            if merger:
                self.merger_occurred = True
                self._merger_step = step
                
                break  # Exit the loop early
            
            # Update velocities
            self.v1 += a1 * self.dt
            self.v2 += a2 * self.dt

            # Update positions
            self.r1 += self.v1 * self.dt
            self.r2 += self.v2 * self.dt

            #Update separation distance and the time at this distance; these values should be saved/updated every iteration as distance decreases, UNTIL distance starts to increase again. The last values saved will be the closest approach distance and time.
            if self.separation_distance > compute_distance(r1, r2):
                self.separation_distance = compute_distance(r1, r2)
                self.separation_time = _t

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

    def save_data(self, filename):
        data = np.column_stack(
            (self.r1_array, self.r2_array, self.r1_array_2d, self.r2_array_2d)
        )
        np.savetxt(filename, data, header=f"merger_occurred={int(self.merger_occurred)}")

    def load_data(self, filename):
        # Read the merger flag from the header line
        with open(filename, "r") as f:
            header = f.readline()  # e.g. "# merger_occurred=1"
            self.merger_occurred = bool(int(header.strip().split("=")[1]))

        data = np.loadtxt(filename)
        self.r1_array = data[:, :3]
        self.r2_array = data[:, 3:6]
        self.r1_array_2d = data[:, 6:8]
        self.r2_array_2d = data[:, 8:]
