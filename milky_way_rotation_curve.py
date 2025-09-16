import pandas as pd
import numpy as np
import os

# --- Paths ---
obs_file = "/sdcard/Download/tab_rcmw.dat"
model_file = "/sdcard/Download/tab_rcmw-model.dat"
output_file = "/sdcard/Download/milky_way_rotation_combined.csv"

# --- Constants ---
kpc_to_m = 3.086e19
kms_to_ms = 1e3

def parse_obs(filename):
    """Parse observed Sofue file with 6-column triplets"""
    raw = pd.read_csv(filename, delim_whitespace=True, header=None, comment='#', engine='python')
    if raw.shape[1] > 6:
        raw = raw.iloc[:, :6]
    if raw.shape[1] < 6:
        raw = raw.reindex(columns=range(6))
    raw = raw.apply(pd.to_numeric, errors='coerce').dropna(how='all').reset_index(drop=True)

    r_mean = raw[[0,1,2]].mean(axis=1)
    v_triplet = raw[[3,4,5]]
    v_mean = v_triplet.mean(axis=1)
    v_min, v_max = v_triplet.min(axis=1), v_triplet.max(axis=1)
    sigma_km_s = ((v_max - v_min)/2.0).fillna(0.0).replace(0,0.5) # 0.5 km/s floor

    df = pd.DataFrame({
        "r_kpc": r_mean,
        "v_km_s": v_mean,
        "sigma_km_s": sigma_km_s,
        "source": "observed"
    })
    return df.dropna(subset=["r_kpc","v_km_s"])

def parse_model(filename):
    """Parse model file: usually 2 columns: r(kpc) v(km/s)"""
    raw = pd.read_csv(filename, delim_whitespace=True, header=None, comment='#', engine='python')
    raw = raw.apply(pd.to_numeric, errors='coerce').dropna(how='all').reset_index(drop=True)

    if raw.shape[1] >= 2:
        r = raw.iloc[:,0]
        v = raw.iloc[:,1]
    else:
        raise ValueError("Model file has unexpected format")

    df = pd.DataFrame({
        "r_kpc": r,
        "v_km_s": v,
        "sigma_km_s": np.nan,
        "source": "model"
    })
    return df

# --- Parse both ---
df_obs = parse_obs(obs_file)
df_model = parse_model(model_file)

# --- Combine ---
df = pd.concat([df_obs, df_model], ignore_index=True)

# --- Unit conversions ---
df["r_m"] = df["r_kpc"] * kpc_to_m
df["v_m_s"] = df["v_km_s"] * kms_to_ms
df["sigma_m_s"] = df["sigma_km_s"] * kms_to_ms

# --- Save ---
save_cols = ["r_kpc","v_km_s","sigma_km_s","r_m","v_m_s","sigma_m_s","source"]
df.to_csv(output_file, columns=save_cols, index=False)

print(f"Saved combined rotation data to: {output_file}")
print(f"Rows written: {len(df)}")
print(df.head(10).to_string(index=False))