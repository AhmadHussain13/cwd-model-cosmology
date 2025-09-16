# ngc3198_plot.py
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Config (edit only if necessary) ----------
FILENAME = "ngc3198_rotation.csv"   # name of your file in file manager
SEARCH_DIRS = [
    ".",  # current working directory
    "/storage/emulated/0/Download",
    "/sdcard/Download",
    "/storage/emulated/0/Documents",
    "/sdcard",
    "/storage/emulated/0"
]
SAVE_DIR = "/storage/emulated/0/Download/astro_figures"
os.makedirs(SAVE_DIR, exist_ok=True)
# -----------------------------------------------------

def find_file(name, search_dirs):
    """Return first existing path for filename in search_dirs (including subdirs)."""
    for d in search_dirs:
        if not d:
            continue
        path = os.path.join(d, name)
        if os.path.exists(path):
            return path
        # try glob searching inside dir
        matches = glob.glob(os.path.join(d, "**", name), recursive=True)
        if matches:
            return matches[0]
    # fallback: search entire sdcard (may be slow)
    for d in search_dirs:
        try:
            matches = glob.glob(os.path.join(d, "**", name), recursive=True)
            if matches:
                return matches[0]
        except Exception:
            pass
    return None

def smart_read_csv(path):
    """Try pandas read_csv with different separators until success."""
    # try automatic engine detection
    try:
        df = pd.read_csv(path, sep=None, engine='python', comment="#")
        return df
    except Exception:
        pass
    # common fallbacks
    for sep in [",", "\t", r"\s+", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, engine='python', comment="#")
            return df
        except Exception:
            continue
    # final fallback: raw read and try to parse manually
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    # try comma-splitting of first non-comment line
    if lines:
        first = lines[0]
        parts = first.split()
        # construct DataFrame assuming whitespace
        try:
            df = pd.read_table(path, delim_whitespace=True, comment="#", engine='python')
            return df
        except Exception:
            pass
    raise RuntimeError("Could not parse CSV/TXT file automatically. Please inspect file format.")

def find_column(df, candidates):
    """Find first column in df whose name matches any candidate (case-insensitive, substring)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cand_low = cand.lower()
        # exact match
        if cand_low in cols_lower:
            return cols_lower[cand_low]
    # substring search
    for col in df.columns:
        cl = col.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return col
    # regex-like alternatives: try common tokens
    for col in df.columns:
        cl = col.lower()
        if any(tok in cl for tok in candidates):
            return col
    return None

def main():
    path = find_file(FILENAME, SEARCH_DIRS)
    if path is None:
        print("ERROR: Could not find", FILENAME, "in common locations. Please move it to Downloads or current dir.")
        print("Searched:", SEARCH_DIRS)
        sys.exit(1)
    print("Loading file:", path)

    df = smart_read_csv(path)
    print("Columns found:", list(df.columns))

    # detect radius (kpc)
    radius_col = find_column(df, ["r_kpc", "radius", "r_kpc", "r"])
    # detect observed velocity (km/s)
    vobs_col = find_column(df, ["v_km_s", "v_obs", "vobs", "v_obs_km", "v"])
    # detect sigma (km/s)
    sigma_col = find_column(df, ["sigma_km_s", "sigma", "err vobs", "err_vobs", "err"])
    # detect model velocity: might be in m/s column 'v_m_s' or 'v_m_s' (m/s) or 'Vt' or 'v_model'
    vmodel_col = find_column(df, ["v_m_s", "v_model", "Vt", "v_model_km", "v_m"])

    if radius_col is None or vobs_col is None:
        print("ERROR: Could not find required columns.")
        print("Required: radius (e.g., r_kpc) and observed velocity (e.g., v_km_s).")
        print("Columns available:", list(df.columns))
        sys.exit(1)

    # Extract arrays; coerce to numeric
    radius = pd.to_numeric(df[radius_col], errors='coerce').to_numpy(dtype=float)
    vobs = pd.to_numeric(df[vobs_col], errors='coerce').to_numpy(dtype=float)
    if sigma_col is not None:
        sigma = pd.to_numeric(df[sigma_col], errors='coerce').to_numpy(dtype=float)
    else:
        sigma = None

    # If observed velocities look like m/s (very large numbers), try to detect and convert to km/s
    # But user file shows v_km_s -> values ~ 5-50 so we assume it's km/s already; detect if vobs >> 1000
    if np.nanmax(np.abs(vobs[np.isfinite(vobs)])) > 1000:
        print("Detected very large velocity values in obs (likely m/s). Converting to km/s by dividing by 1000.")
        vobs = vobs / 1000.0
        if sigma is not None and np.nanmax(np.abs(sigma[np.isfinite(sigma)])) > 1000:
            sigma = sigma / 1000.0

    # Handle model column if present
    vmodel_kms = None
    if vmodel_col is not None:
        vmodel = pd.to_numeric(df[vmodel_col], errors='coerce').to_numpy(dtype=float)
        # if values are huge assume m/s -> km/s
        if np.nanmax(np.abs(vmodel[np.isfinite(vmodel)])) > 1000:
            vmodel_kms = vmodel / 1000.0
        else:
            vmodel_kms = vmodel.copy()

    # Remove NaNs for plotting
    good = np.isfinite(radius) & np.isfinite(vobs)
    if sigma is not None:
        good = good & np.isfinite(sigma)
    if not np.any(good):
        print("ERROR: No valid numeric rows found after parsing.")
        sys.exit(1)

    radius_plot = radius[good]
    vobs_plot = vobs[good]
    sigma_plot = sigma[good] if sigma is not None else None
    # sort by radius for nicer plotting
    order = np.argsort(radius_plot)
    radius_plot = radius_plot[order]
    vobs_plot = vobs_plot[order]
    if sigma_plot is not None:
        sigma_plot = sigma_plot[order]

    # Plot
    plt.figure(figsize=(8,5))
    if sigma_plot is not None:
        plt.errorbar(radius_plot, vobs_plot, yerr=sigma_plot, fmt='o', ecolor='gray', capsize=3, label="Observed")
    else:
        plt.plot(radius_plot, vobs_plot, 'o', label="Observed")

    if vmodel_kms is not None:
        vmodel_plot = vmodel_kms[good][order]
        plt.plot(radius_plot, vmodel_plot, '-', label="Model (from file)")

    plt.xlabel("Radius (kpc)")
    plt.ylabel("Velocity (km/s)")
    plt.title("NGC 3198 Rotation Curve")
    plt.grid(True)
    plt.legend()

    outpath = os.path.join(SAVE_DIR, "ngc3198_rotation_curve.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()

    print("Plot saved to:", outpath)
    print("Used columns -> radius:", radius_col, ", vobs:", vobs_col, ", sigma:", sigma_col, ", vmodel:", vmodel_col)

if __name__ == "__main__":
    main()