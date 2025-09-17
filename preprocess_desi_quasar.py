# scripts/preprocess_desi_quasar.py
"""
Preprocess (mock) DESI quasar data for Appendix F.
- Generates synthetic P_quasar(k) at z ~ 3
- Adds Gaussian noise
- Saves to data/desi_quasar_pk_z3.txt
"""

import numpy as np
import os

# -----------------------------
# Settings
# -----------------------------
outdir = "data"
outfile = os.path.join(outdir, "desi_quasar_pk_z3.txt")

# Wavenumbers: logarithmic bins between 0.01 and 0.5 h/Mpc
k = np.logspace(-2, -0.3, 25)  # ~25 points

# True underlying "shape" for mock spectrum (ΛCDM-like decline)
P_true = 1e4 * (k / 0.1) ** -1.3 * np.exp(-k / 0.5)

# Noise model: 5% relative error
rng = np.random.default_rng(seed=42)
noise_frac = 0.05
sigma = noise_frac * P_true
P_obs = P_true + rng.normal(0, sigma)

# Covariance: just diagonal for mock
cov_diag = sigma**2

# Stack into table
table = np.column_stack([k, P_obs, cov_diag])

# Ensure directory exists
os.makedirs(outdir, exist_ok=True)

# Save file
np.savetxt(
    outfile,
    table,
    header="k[h/Mpc]   P_obs[(Mpc/h)^3]   diag_cov",
    fmt="%.6e"
)

print(f"Mock DESI quasar data written to {outfile}")