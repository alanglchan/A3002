# AlanC 20240922 Analytical solutions for different Universe type at redshift of 0.01, 0.1, and 1
# V2.2
# fixed km/s from m/s

import numpy as np

# Constants
H_0 = 70  # Hubble constant in km/s/Mpc
c = 299792.458  # Speed of light in km/s

# Convert Hubble constant to 1/Gyr (1 Gyr = 3.086e19 km)
H_0 = H_0 / 3.086e19  # H_0 in 1/Gyr
c_Mpc = c / 3.086e19  # Speed of light in Mpc/Gyr

# Analytical solutions for different universes
def empty_universe_dp(z):
    return c_Mpc / H_0 * np.log(1 + z)

def matter_only_dp(z):
    return 2 * c_Mpc / H_0 * (1 - (1 + z)**(-1/2))

def lambda_only_dp(z):
    return c_Mpc * z / H_0

# Function to calculate DA, DP, and DL
def calculate_distances(z, dp_function):
    D_P = dp_function(z)
    D_A = D_P / (1 + z)
    D_L = D_P * (1 + z)
    return D_P, D_A, D_L

# Redshifts of interest
z_values = [0.01, 0.1, 1]

# Empty universe
print("Empty Universe:")
for z in z_values:
    D_P, D_A, D_L = calculate_distances(z, empty_universe_dp)
    print(f"z = {z}: D_P = {D_P:.6f} Mpc, D_A = {D_A:.6f} Mpc, D_L = {D_L:.6f} Mpc")

# Matter-only universe
print("\nMatter-Only Universe:")
for z in z_values:
    D_P, D_A, D_L = calculate_distances(z, matter_only_dp)
    print(f"z = {z}: D_P = {D_P:.6f} Mpc, D_A = {D_A:.6f} Mpc, D_L = {D_L:.6f} Mpc")

# Lambda-only universe
print("\nLambda-Only Universe:")
for z in z_values:
    D_P, D_A, D_L = calculate_distances(z, lambda_only_dp)
    print(f"z = {z}: D_P = {D_P:.6f} Mpc, D_A = {D_A:.6f} Mpc, D_L = {D_L:.6f} Mpc")
