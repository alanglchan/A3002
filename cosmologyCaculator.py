#Author: Alan Chan (alanc@mso.anu.edu.au)
#20240911 - for A3002 Q1

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import astropy.constants as const
import astropy.units as u

# Cosmological Parameters
Omega_Lambda = 0.7
Omega_m = 0.3
Omega_k = 1.0 - (Omega_Lambda + Omega_m)
H_0 = 70e-3  # Hubble constant in km/s/Mpc (converted to Gyr^-1 for simplicity)
epsilon = 1e-10  # Small value to avoid negative sqrt

# Friedmann Equation (scale factor evolution)
def friedmann_eq(a, t):    
    term = Omega_m/a**3 + Omega_Lambda + Omega_k/a**2
    term = np.maximum(term, epsilon)  # Prevent negative values inside sqrt
    H = H_0 * np.sqrt(term)
    return -H * a

# Function to calculate all quantities for a given dt
def calculate_quantities(dt):
    t = np.arange(0, 13.78, dt)
    a0 = 1.0  # Present scale factor
    a = odeint(friedmann_eq, a0, t, rtol=1e-5, atol=1e-8).flatten()  # Solve for a(t)

    # Mask NaN values
    valid_mask = ~np.isnan(a)
    a = a[valid_mask]
    t = t[valid_mask]

    # Redshift z = 1/a - 1
    z = 1/a - 1

    # Proper distance calculation
    speed_of_light = const.c.to(u.Glyr/u.Gyr).value  # c in Glyr/Gyr
    dp = speed_of_light * np.cumsum(1/a) * dt  # Proper distance in Glyr

    # Curvature and Angular Diameter Distance
    if Omega_k > 0:
        R_0 = speed_of_light / (H_0 * np.sqrt(Omega_k))
        d_A = R_0 * np.sinh(dp/R_0) / (1 + z)
    elif Omega_k < 0:
        R_0 = speed_of_light / (H_0 * np.sqrt(-Omega_k))
        d_A = R_0 * np.sin(dp/R_0) / (1 + z)
    else:
        d_A = dp / (1 + z)

    # Luminosity Distance
    d_L = dp * (1 + z)

    # Hubble Parameter
    H = H_0 * np.sqrt(Omega_m/a**3 + Omega_Lambda + Omega_k/a**2) * 1e3  # km/s/Mpc

    # Distance Modulus
    dist_mod = 5 * np.log10(dp * u.Glyr.to(u.pc) / 10)

    return t, a, z, dp, d_A, d_L, H, dist_mod

# Plotting function for multiple dt values with different line styles
def plot_multiple_dt(dts, title, y_label, calc_func):
    plt.figure(figsize=(8, 6))

    # Define line width, alpha (transparency), and line styles based on dt
    line_widths = {1.0: 2, 0.1: 2, 0.01: 2}
    alphas = {1.0: 1, 0.1: 0.8, 0.01: 0.7}
    line_styles = {1.0: '-', 0.1: '--', 0.01: '-.'}  # Solid, dashed, dash-dot

    for dt in dts:
        t, y_values = calc_func(dt)
        plt.plot(t, y_values, label=f'dt = {dt}', linewidth=line_widths.get(dt, 1), 
                 alpha=alphas.get(dt, 1), linestyle=line_styles.get(dt, '-'))

    plt.title(title)
    plt.xlabel(r'$\tau$ (Gyr)')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()  # Add legend to differentiate the dt values
    plt.show()

# Functions to return specific plots
def plot_scale_factor(dts):
    plot_multiple_dt(dts, r'Scale Factor a($\tau$) vs Time $\tau$', r'a($\tau$)', lambda dt: (calculate_quantities(dt)[:2]))

def plot_redshift(dts):
    plot_multiple_dt(dts, r'Redshift z($\tau$) vs Time $\tau$', r'z($\tau$)', lambda dt: (calculate_quantities(dt)[0], calculate_quantities(dt)[2]))

def plot_proper_distance(dts):
    plot_multiple_dt(dts, r'Proper Distance $d_p$($\tau$)', r'$d_p$ (Glyr)', lambda dt: (calculate_quantities(dt)[0], calculate_quantities(dt)[3]))

def plot_angular_diameter_distance(dts):
    plot_multiple_dt(dts, r'Angular Diameter Distance $d_A$($\tau$)', r'$d_A$ (Glyr)', lambda dt: (calculate_quantities(dt)[0], calculate_quantities(dt)[4]))

def plot_luminosity_distance(dts):
    plot_multiple_dt(dts, r'Luminosity Distance $d_L$($\tau$)', r'$d_L$ (Glyr)', lambda dt: (calculate_quantities(dt)[0], calculate_quantities(dt)[5]))

def plot_hubble_parameter(dts):
    plot_multiple_dt(dts, r'Hubble Parameter H($\tau$) vs $\tau$', 'H (km/s/Mpc)', lambda dt: (calculate_quantities(dt)[0], calculate_quantities(dt)[6]))

def plot_distance_modulus(dts):
    plot_multiple_dt(dts, r'Distance Modulus $\mu$ vs $\tau$', r'$\mu$', lambda dt: (calculate_quantities(dt)[0], calculate_quantities(dt)[7]))

# List of dt values to plot
dts = [1.0, 0.1, 0.01]

# Call the plotting functions
plot_scale_factor(dts)
plot_redshift(dts)
plot_proper_distance(dts)
plot_angular_diameter_distance(dts)
plot_luminosity_distance(dts)
plot_hubble_parameter(dts)
plot_distance_modulus(dts)
