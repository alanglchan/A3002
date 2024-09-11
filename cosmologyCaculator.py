#Author: Alan Chan (alanc@mso.anu.edu.au)
#20240911 - for A3002 Q1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# Constants
H0 = 70  # Hubble constant in km/s/Mpc
c = 3e5  # Speed of light in km/s

# Different cosmological models (ensure all return arrays)
def hubble_empty_universe(z):
    """Hubble parameter for an empty universe."""
    return H0 * (1 + z)

def hubble_matter_only_universe(z):
    """Hubble parameter for a matter-only universe."""
    Omega_m = 1.0
    return H0 * np.sqrt(Omega_m * (1 + z)**3)

def hubble_lambda_only_universe(z):
    """Hubble parameter for a lambda-only universe."""
    Omega_lambda = 1.0
    return H0 * np.sqrt(Omega_lambda * np.ones_like(z))

def hubble_mixed_universe(z):
    """Hubble parameter for a universe with Ω_m = 0.3, Ω_Λ = 0.7."""
    Omega_m = 0.3
    Omega_lambda = 0.7
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_lambda)

# Luminosity distance
def luminosity_distance(z, hubble_fn):
    """Calculate luminosity distance D_L for a given hubble function."""
    z_vals = np.linspace(0, z, 1000)  # Create an array of redshift values
    integrand = c / hubble_fn(z_vals)  # Vectorized operation
    D_C = np.trapz(integrand, z_vals)  # Perform numerical integration
    return D_C * (1 + z)

# Calculate distance modulus μ
def distance_modulus(D_L):
    """Calculate the distance modulus μ."""
    return 5 * np.log10(D_L) - 5

# Apply the model to data
def calculate_moduli(data, hubble_fn):
    """Calculate the distance moduli for the given data and Hubble function."""
    mu_model = []
    for z in data['z']:
        D_L = luminosity_distance(z, hubble_fn)
        mu_model.append(distance_modulus(D_L))
    return mu_model

def main():
    # Parse the CSV file path from the command line
    parser = argparse.ArgumentParser(description="Cosmology Calculator")
    parser.add_argument("csv_file", help="Path to the CSV file containing redshift and distance modulus data")
    args = parser.parse_args()

    # Load data from the provided CSV file and strip whitespace from columns
    data = pd.read_csv(args.csv_file)
    data.columns = data.columns.str.strip()  # Strip any leading/trailing whitespace from column names

    # Display the columns to verify the correct names
    print("Columns in the dataset:", data.columns)

    # Ensure we use the correct column names
    z_column = 'z'        # Column for redshift
    mu_column = 'mu'      # Column for distance modulus
    dmu_column = 'dmu'    # Column for error in distance modulus

    # Compare with four models
    data['mu_empty'] = calculate_moduli(data, hubble_empty_universe)
    data['mu_matter_only'] = calculate_moduli(data, hubble_matter_only_universe)
    data['mu_lambda_only'] = calculate_moduli(data, hubble_lambda_only_universe)
    data['mu_mixed'] = calculate_moduli(data, hubble_mixed_universe)

    # Plot the distance modulus vs redshift
    plt.figure(figsize=(10, 6))
    plt.errorbar(data[z_column], data[mu_column], yerr=data[dmu_column], fmt='o', label="Observed Data", capsize=1)
    plt.plot(data[z_column], data['mu_empty'], label="Empty Universe")
    plt.plot(data[z_column], data['mu_matter_only'], label="Matter-only Universe")
    plt.plot(data[z_column], data['mu_lambda_only'], label="Lambda-only Universe")
    plt.plot(data[z_column], data['mu_mixed'], label="Mixed Universe (Ω_m=0.3, Ω_Λ=0.7)")

    plt.xlabel('Redshift (z)')
    plt.ylabel('Distance Modulus (μ)')
    plt.title('Distance Modulus vs Redshift for Different Cosmological Models')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
