import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# USER EDIT: adjust paths
# -------------------------
file_path = "/storage/emulated/0/Download/ngc3198.dat"   # path to your file
output_path = "/storage/emulated/0/Download/ngc3198_rotation.csv"
# -------------------------

# Helper: show first few raw lines (for debugging)
with open(file_path, 'r', errors='ignore') as f:
    raw_lines = [next(f) for _ in range(6)]
print("First 6 lines of file (raw):")
for i, line in enumerate(raw_lines, 1):
    print(f"{i:02d}: {line.rstrip()}")

# Read the file robustly:
# - delim_whitespace=True handles variable spacing (common in .dat)
# - comment="#" ignores lines that start with '#' (units lines)
# - header=0 uses the first non-comment line as column names
df = pd.read_csv(file_path,
                 delim_whitespace=True,
                 comment='#',
                 header=0,
                 skipinitialspace=True,
                 engine='python')  # engine=python is slightly more permissive

print("\nColumns found:", df.columns.tolist())

# Drop completely empty 'Unnamed...' columns if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Clean column names: strip spaces, lower, replace spaces with underscores
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

print("Normalized columns:", df.columns.tolist())

# Try to find which column is the observed velocity & its error
# Common names in your file: 'vobs', 'err vobs' -> normalized to 'vobs', 'err_vobs'
possible_vcols = [c for c in df.columns if 'vobs' in c or 'v_obs' in c or c in ('v','v_obs')]
possible_errcols = [c for c in df.columns if 'err' in c and ('v' in c or 'vel' in c)]

print("Candidate velocity columns:", possible_vcols)
print("Candidate error columns:", possible_errcols)

# If not obvious, list column head to inspect
print("\nFirst rows (head):")
print(df.head())

# Coerce all columns to numeric where possible (non-numeric -> NaN)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where radius is NaN, because they're useless
radius_col = None
for candidate in ['radius', 'r', 'r_kpc', '#', 'radius_kpc']:
    if candidate in df.columns:
        radius_col = candidate
        break
if radius_col is None:
    # try to find numeric column with increasing values typical of radius
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # pick the first numeric column as radius (best effort)
    if len(numeric_cols) > 0:
        radius_col = numeric_cols[0]
        print(f"Warning: No explicit 'radius' column found; using '{radius_col}' as radius.")
    else:
        raise RuntimeError("Could not auto-detect a radius column. Inspect file manually.")

# Choose velocity column (vobs preferred)
if 'vobs' in df.columns:
    vcol = 'vobs'
elif possible_vcols:
    vcol = possible_vcols[0]
else:
    # fallback: try 'v' or the next numeric column after radius
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    try:
        ridx = numeric_cols.index(radius_col)
        vcol = numeric_cols[ridx + 1]
        print(f"Warning: falling back to numeric column '{vcol}' for velocities.")
    except Exception:
        raise RuntimeError("Could not auto-detect velocity column. Inspect file manually.")

# Choose error column
if 'err_vobs' in df.columns:
    errcol = 'err_vobs'
elif possible_errcols:
    errcol = possible_errcols[0]
else:
    # fallback: try to find a column containing 'err' or the next numeric column after vcol
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    try:
        vidx = numeric_cols.index(vcol)
        errcol = numeric_cols[vidx + 1]
        print(f"Warning: falling back to numeric column '{errcol}' for velocity errors.")
    except Exception:
        errcol = None
        print("No error column auto-detected; will estimate errors from scatter if needed.")

print(f"\nUsing radius column: {radius_col}")
print(f"Using velocity column: {vcol}")
print(f"Using error column: {errcol}")

# Keep only rows with numeric radius and velocity
keep_mask = df[radius_col].notna() & df[vcol].notna()
df_clean = df.loc[keep_mask, [radius_col, vcol] + ([errcol] if errcol else [])].copy()

# If velocities are signed and seem to be negative (like some formats), take absolute
df_clean[vcol] = df_clean[vcol].abs()

# If no error column, estimate an error: e.g., 5% of v or a floor (you can change as needed)
if errcol is None:
    df_clean['sigma_km_s'] = (0.05 * df_clean[vcol]).replace(0, 1.0)  # 5% fallback
    errcol_out = 'sigma_km_s'
else:
    df_clean['sigma_km_s'] = df_clean[errcol].abs()
    errcol_out = 'sigma_km_s'

# Rename radius -> r_kpc and velocity -> v_km_s for consistency
df_clean = df_clean.rename(columns={radius_col: 'r_kpc', vcol: 'v_km_s'})

# Print some stats
print("\nCleaned data sample:")
print(df_clean.head())

# -------------------------
# Bin every 2 kpc
# -------------------------
r_min = max(0.0, df_clean['r_kpc'].min())
r_max = df_clean['r_kpc'].max()
bin_size = 2.0  # kpc, per your appendix
bins = np.arange(r_min, r_max + bin_size, bin_size)

binned_rows = []
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (df_clean['r_kpc'] >= lo) & (df_clean['r_kpc'] < hi)
    if not mask.any():
        continue
    r_mean = df_clean.loc[mask, 'r_kpc'].mean()
    v_mean = df_clean.loc[mask, 'v_km_s'].mean()
    # combine errors in quadrature averaged (RMS):
    sigma_rms = np.sqrt((df_clean.loc[mask, 'sigma_km_s']**2).mean())
    binned_rows.append((r_mean, v_mean, sigma_rms))

# If no bins found, try a coarser bin (safety)
if len(binned_rows) == 0:
    print("No bins produced with 2 kpc step — trying single-bin fallback (all data averaged).")
    r_mean = df_clean['r_kpc'].mean()
    v_mean = df_clean['v_km_s'].mean()
    sigma_rms = np.sqrt((df_clean['sigma_km_s']**2).mean())
    binned_rows.append((r_mean, v_mean, sigma_rms))

binned_df = pd.DataFrame(binned_rows, columns=['r_kpc', 'v_km_s', 'sigma_km_s'])

# Convert to SI
kpc_to_m = 3.086e19   # 1 kpc in meters (approximate)
binned_df['r_m'] = binned_df['r_kpc'] * kpc_to_m
binned_df['v_m_s'] = binned_df['v_km_s'] * 1e3
binned_df['sigma_m_s'] = binned_df['sigma_km_s'] * 1e3

# Save
binned_df = binned_df[['r_kpc','r_m','v_km_s','v_m_s','sigma_km_s','sigma_m_s']]
out_dir = os.path.dirname(output_path)
if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)
binned_df.to_csv(output_path, index=False)
print(f"\nBinned CSV saved to: {output_path}")
print("\nBinned data preview:")
print(binned_df.head(20))

# Plot for quick check
plt.errorbar(binned_df['r_kpc'], binned_df['v_km_s'], yerr=binned_df['sigma_km_s'],
             fmt='o', markersize=6, ecolor='gray', capsize=3)
plt.xlabel("Radius (kpc)")
plt.ylabel("Velocity (km/s)")
plt.title("NGC 3198 — Binned Rotation Curve (2 kpc bins)")
plt.grid(True)
plt.show()