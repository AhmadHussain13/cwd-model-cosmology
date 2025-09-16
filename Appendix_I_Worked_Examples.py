#!/usr/bin/env python3
# appendixE_examples_with_tables.py
"""
Appendix E — Worked Examples in SI Units
----------------------------------------
This script reproduces arithmetic from Appendix E and now also outputs tables:
- Draco, Milky Way, NGC 3198
- Step-by-step printed numbers
- Figures: draco_example.png, milkyway_example.png, ngc3198_example.png
- CSV tables: table_E1_inputs.csv, table_E2_outputs.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
G = 6.67430e-11          # m^3 kg^-1 s^-2
Msun = 1.98847e30        # kg
kpc = 3.085677581e19     # m
kms = 1.0e3              # 1 km/s in m/s

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def v2_pointmass(M, r):
    return G * M / r

def v2_yukawa_like(M5, r, L):
    x = r / L
    return (G * M5 / r) * np.exp(-x) * (1 + x + x**2)

# ----------------------------------------------------------------------
# Storage for tables
# ----------------------------------------------------------------------
table_E1 = []  # Inputs
table_E2 = []  # Outputs

# ----------------------------------------------------------------------
# Draco example
# ----------------------------------------------------------------------
def draco_example():
    print("\n--- Draco Example ---")
    M_baryon = 6e35  # kg
    r = 1.0 * kpc
    L = 10.0 * kpc
    sigma_obs = 10 * kms

    # Store inputs
    table_E1.append(["Draco", M_baryon, r/kpc, L/kpc, sigma_obs/kms])

    # 1) Baryonic
    v2_baryon = v2_pointmass(M_baryon, r)
    v_baryon = np.sqrt(v2_baryon)

    # 2) 5D coefficient
    C = (G / r) * np.exp(-r/L) * (1 + r/L + (r/L)**2)

    # 3) Solve for M5 using isotropic mapping
    v_total = np.sqrt(3) * sigma_obs
    v2_total = v_total**2
    v2_5D = v2_total - v2_baryon
    M5 = v2_5D / C
    alpha = M5 / M_baryon

    print(f"Baryonic v^2 = {v2_baryon:.3e} m^2/s^2, v = {v_baryon/1e3:.2f} km/s")
    print(f"Coefficient C = {C:.3e}")
    print(f"Target v_total = {v_total/1e3:.2f} km/s")
    print(f"v^2_5D needed = {v2_5D:.3e}")
    print(f"Solved M5 = {M5:.3e} kg")
    print(f"Alpha = {alpha:.2f}")

    # Store outputs
    table_E2.append(["Draco", v2_baryon, v_baryon, C, M5, alpha])

    # Plot
    r_grid = np.linspace(0.1, 2.0, 200) * kpc
    v_baryon_grid = np.sqrt(v2_pointmass(M_baryon, r_grid))
    v_5D_grid = np.sqrt(v2_yukawa_like(M5, r_grid, L))
    v_tot_grid = np.sqrt(v_baryon_grid**2 + v_5D_grid**2)

    plt.figure(figsize=(8,6))
    plt.plot(r_grid/kpc, v_baryon_grid/1e3, label="Baryons only")
    plt.plot(r_grid/kpc, v_tot_grid/1e3, label="Baryons + 5D")
    plt.axhline(sigma_obs/1e3, color="black", ls="--", label="Observed σ")
    plt.xlabel("Radius [kpc]")
    plt.ylabel("Velocity [km/s]")
    plt.title("Draco worked example")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("draco_example.png", dpi=150)
    plt.close()
    print("Saved draco_example.png")

# ----------------------------------------------------------------------
# Milky Way example
# ----------------------------------------------------------------------
def milkyway_example():
    print("\n--- Milky Way Example ---")
    M_baryon = 6e10 * Msun
    r = 8.0 * kpc
    v_obs = 220 * kms

    table_E1.append(["Milky Way", M_baryon, r/kpc, "-", v_obs/kms])

    v2_baryon = v2_pointmass(M_baryon, r)
    v_baryon = np.sqrt(v2_baryon)
    v2_obs = v_obs**2
    deficit = v2_obs - v2_baryon

    print(f"Baryon v = {v_baryon/1e3:.2f} km/s at 8 kpc")
    print(f"Observed v = {v_obs/1e3:.2f} km/s")
    print(f"Deficit in v^2 = {deficit:.3e} m^2/s^2")

    table_E2.append(["Milky Way", v2_baryon, v_baryon, "-", "-", "-"])

    plt.figure(figsize=(8,6))
    plt.axhline(v_obs/1e3, color="black", ls="--", label="Observed ~220 km/s")
    plt.plot([r/kpc], [v_baryon/1e3], "ro", label="Baryons (point-mass)")
    plt.xlabel("Radius [kpc]")
    plt.ylabel("Velocity [km/s]")
    plt.title("Milky Way simple check at 8 kpc")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("milkyway_example.png", dpi=150)
    plt.close()
    print("Saved milkyway_example.png")

# ----------------------------------------------------------------------
# NGC 3198 example
# ----------------------------------------------------------------------
def ngc3198_example():
    print("\n--- NGC 3198 Example ---")
    M_baryon = 3e10 * Msun
    r = 15.0 * kpc
    v_obs = 150 * kms

    table_E1.append(["NGC 3198", M_baryon, r/kpc, "-", v_obs/kms])

    v2_baryon = v2_pointmass(M_baryon, r)
    v_baryon = np.sqrt(v2_baryon)
    v2_obs = v_obs**2
    deficit = v2_obs - v2_baryon

    print(f"Baryon v = {v_baryon/1e3:.2f} km/s at 15 kpc")
    print(f"Observed v = {v_obs/1e3:.2f} km/s")
    print(f"Deficit in v^2 = {deficit:.3e} m^2/s^2")

    table_E2.append(["NGC 3198", v2_baryon, v_baryon, "-", "-", "-"])

    plt.figure(figsize=(8,6))
    plt.axhline(v_obs/1e3, color="black", ls="--", label="Observed ~150 km/s")
    plt.plot([r/kpc], [v_baryon/1e3], "ro", label="Baryons (point-mass)")
    plt.xlabel("Radius [kpc]")
    plt.ylabel("Velocity [km/s]")
    plt.title("NGC 3198 simple check at 15 kpc")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("ngc3198_example.png", dpi=150)
    plt.close()
    print("Saved ngc3198_example.png")

# ----------------------------------------------------------------------
# Save tables
# ----------------------------------------------------------------------
def save_tables():
    # Example: save to Downloads folder on Android
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(downloads_path, exist_ok=True)  # make sure it exists

    table1_file = os.path.join(downloads_path, "table_E1_inputs.csv")
    table2_file = os.path.join(downloads_path, "table_E2_outputs.csv")

    with open(table1_file, "w", newline="") as f1:
        writer = csv.writer(f1)
        writer.writerow(["Object", "M_baryon [kg]", "r [kpc]", "L [kpc]", "Observed v or σ [km/s]"])
        writer.writerows(table_E1)

    with open(table2_file, "w", newline="") as f2:
        writer = csv.writer(f2)
        writer.writerow(["Object", "v_baryon^2 [m^2/s^2]", "v_baryon [km/s]", "C", "M5 [kg]", "Alpha"])
        writer.writerows(table_E2)

    print(f"Saved tables to {downloads_path}")


# ----------------------------------------------------------------------
# Run all
# ----------------------------------------------------------------------
if __name__ == "__main__":
    draco_example()
    milkyway_example()
    ngc3198_example()
    save_tables()
    print("\nAll Appendix E examples done.")