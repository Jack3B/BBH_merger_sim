from __future__ import annotations
import numpy as np

G = 6.67430e-11  # Gravitational constant, in m^3 / (kg s^2)
c = 2.99792458e8  # Speed of light, in m/s

def convert_kilogram_solar_mass(mass): #Convert from kilograms to solar masses
    return (mass * 5.02785) * (10 ** -31)

def convert_solar_mass_kilogram(mass): #Convert from solar masses to kilograms
    return (mass * 1.989) * (10 ** 30)

def convert_meter_au(distance): #Convert from meters to AU
    return (distance * 6.68459) * (10 ** -12)

def convert_au_meter(distance): #Convert from AU to meters
    return (distance * 1.496) * (10 ** 11)

def compute_unit_vector(vector): #Gets the unit vectors for both the x/y components of the input vector
    vector_mag = np.linalg.norm(vector)
    if vector_mag == 0:
        return 0.0, 0.0
    unit_x_vector = vector[0] / vector_mag
    unit_y_vector = vector[1] / vector_mag
    return unit_x_vector, unit_y_vector

def compute_acceleration(r, v, m1, m2, pn_order=1, radiation=False, spins=None): #Finds global acceleration of the system
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    a_newton = -G * (m1 + m2) / r_mag**3 * r

    a_pn = np.zeros_like(r)
    if pn_order >= 1:
        a_pn += compute_1pn_correction(r, v, r_mag, v_mag, m1, m2)
    if pn_order >= 2:
        a_pn += compute_2pn_correction(r, v, r_mag, v_mag, m1, m2)

    a_rad_reaction = np.zeros_like(r)
    if radiation:
        a_rad_reaction = compute_radiation_reaction(r, v, r_mag, m1, m2)

    a_spin = np.zeros_like(r)
    if spins is not None:
        a_spin = compute_spin_effects(r, v, r_mag, spins)

    return a_newton + a_pn + a_rad_reaction + a_spin

def compute_schwarzschild_radii(m1, m2): #Calculates the Schwarzschild radius in meters for a black hole with mass m in kg
    r_sch1 = (2*G*m1) / (c**2)
    r_sch2 = (2*G*m2) / (c**2)
    
    return(r_sch1, r_sch2)
    

def compute_merger_event_test(r1, r2, r_sch1, r_sch2): #if separation distance <= r_sch1 + r_sch2 then merger == true)
    x1, y1 = r1
    x2, y2 = r2
    
    d_sep = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    merger = d_sep <= r_sch1 + r_sch2
    
    return merger


def compute_distance(r1, r2): #Returns the distance between the two black holes; this value should be saved/updated every iteration as it decreases, UNTIL it starts to increase again. The last value saved will be the closest approach distance.
    x1, y1 = r1
    x2, y2 = r2
    d_sep = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    return (d_sep)


def compute_remaining_distance_for_merger(separation_distance, r_sch1, r_sch2): #Returns how much closer the black holes needed to get for a merger to occur
    return max(0, separation_distance - (r_sch1 + r_sch2))

def compute_deflection_angle(unit_vector_initial, unit_vector_final): #Returns the deflection angle between an initial and final unit vector, in degrees
    return np.degrees(np.arccos(np.dot(unit_vector_initial, unit_vector_final)))

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
