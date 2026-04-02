# dynamics.py

import numpy as np

G = 6.67430e-11  # Gravitational constant, in m^3 / (kg s^2)
c = 2.99792458e8  # Speed of light, in m/s

def convert_kilogram_solar_mass(mass): #Convert from kilograms to solar masses
    return (mass * 5.02785) * (10 ** -31)

def convert_solar_mass_kilogram(mass): #Convert from solar masses to kilograms
    return (mass * 1.989) * (10 ** 30)

def convert_meter_au(distance): #Convert from meters to AU
    return (distance * 6.68459) * (10 ** -12)

def compute_acceleration(r, v, m1, m2, pn_order=1, radiation=False, spins=None):
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # Newtonian acceleration
    a_newton = -G * (m1 + m2) / r_mag**3 * r

    # Post-Newtonian corrections
    a_pn = np.zeros(3)
    if pn_order >= 1:
        a_pn += compute_1pn_correction(r, v, r_mag, v_mag, m1, m2)
    if pn_order >= 2:
        a_pn += compute_2pn_correction(r, v, r_mag, v_mag, m1, m2)

    # Radiation reaction correction
    a_rad_reaction = np.zeros(3)
    if radiation:
        a_rad_reaction = compute_radiation_reaction(r, v, r_mag, m1, m2)

    # Spin effects (simplified)
    a_spin = np.zeros(3)
    if spins is not None:
        a_spin = compute_spin_effects(r, v, r_mag, spins)

    return a_newton + a_pn + a_rad_reaction + a_spin

def compute_schwarzschild_radii(m1, m2) #Calculates the Schwarzschild radius in meters for a black hole with mass m in kg
    return( 
        r_sch1 = (2*G*m1) / (c**2)
        r_sch2 = (2*G*m2) / (c**2)
    )
    

def compute_merger_event_test(r1, r2, r_sch1, r_sch2): #if separation distance <= r_sch1 + r_sch2 then merger == true)
    x1, y1 = r1
    x2, y2 = r2
    
    d_sep = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    merger = d_sep <= r_sch1 + r_sch2
    
    return merger


def compute_1pn_correction(r, v, r_mag, v_mag, m1, m2):
    return (
        G
        * (m1 + m2)
        / (c**2 * r_mag**2)
        * ((4 * G * (m1 + m2) / r_mag - v_mag**2) * r + 4 * np.dot(r, v) * v)
    )


def compute_2pn_correction(r, v, r_mag, v_mag, m1, m2):
    return (
        G
        * (m1 + m2)
        / (c**4 * r_mag**2)
        * (
            ((2 * G * (m1 + m2) / r_mag) * (2 * v_mag**2 - 9 * G * (m1 + m2) / r_mag))
            * r
            + (v_mag**2 - 3 * G * (m1 + m2) / r_mag) * 4 * np.dot(r, v) * v
            - (3 * G * (m1 + m2) / r_mag)
            * (4 * v_mag**2 - 2 * G * (m1 + m2) / r_mag)
            * r
        )
    )


def compute_radiation_reaction(r, v, r_mag, m1, m2):
    v_dot_r = np.dot(v, r)
    return (
        -32
        / 5
        * (G**3 * m1 * m2 * (m1 + m2))
        / (c**5 * r_mag**4)
        * (v + 3 / 2 * v_dot_r / r_mag * r)
    )


def compute_spin_effects(r, v, r_mag, spins):
    s1, s2 = spins
    return (G / c**2) * (
        2 * np.cross(v, s1) / r_mag**3 + 2 * np.cross(v, s2) / r_mag**3
    )
