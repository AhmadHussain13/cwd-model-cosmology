# likelihood.py
"""
Robust likelihood helpers for the CWD analysis.

Provides:
- try_load_all() -> (mw_df, ngc_df, draco_df_or_None, small_scale_df_or_None)
- load_rotation_for(name)
- find_or_build_draco_sigma()
- log_likelihood_rotation(...)
- log_likelihood_draco_dispersion(...)
- log_likelihood_small_scale(...)
- evaluate_model_velocity(...)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence
import logging
import numpy as np
import pandas as pd

# configure simple logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cwd.likelihood")

# search directories (adjust to your environment as needed)
SEARCH_DIRS = [
    Path("/storage/emulated/0/Download"),
    Path("/storage/emulated/0/data"),
    Path("/storage/emulated/0"),
    Path.cwd(),
]

OUTPUT_DIR = Path("/storage/emulated/0/Download/cwd_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Physical constants & model defaults -----------------
G_4 = 6.67430e-11  # m^3 kg^-1 s^-2
KPC_TO_M = 3.085677581e19
M_SUN = 1.98847e30  # kg
L = 4.629e20  # 15 Mpc in meters (phenomenological Yukawa length)
X0 = 1e-8  # dimensionless cutoff scale applied to (R/L)/X0
GAMMA = 0.48  # mass-scaling exponent (alpha ∝ M^{-gamma})

# ----------------- Basic robust I/O utilities -----------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lower_snake_case-like names."""
    newcols = []
    for c in df.columns:
        s = str(c).strip().lower()
        s = s.replace("#", "").replace(" ", "_").replace("-", "_").replace(".", "_")
        newcols.append(s)
    df.columns = newcols
    return df


def _read_table_try(path: Path) -> pd.DataFrame:
    """Try reading a table with common separators; return normalized columns."""
    if not path.exists():
        raise FileNotFoundError(path)
    # try common separators
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", comment="#", header=0)
            if df.shape[1] > 1:
                return _normalize_columns(df)
        except Exception:
            pass
    # fallback to whitespace-split
    df = pd.read_csv(path, sep=r"\s+", engine="python", comment="#", header=0)
    return _normalize_columns(df)


def _find_file_by_names(names: Sequence[str]) -> Optional[Path]:
    """Search SEARCH_DIRS for exact filenames, then for fuzzy matches."""
    for d in SEARCH_DIRS:
        for n in names:
            p = d / n
            if p.exists():
                return p
    # fuzzy match by stem
    for d in SEARCH_DIRS:
        for n in names:
            stem = Path(n).stem
            for f in d.glob(f"*{stem}*"):
                if f.is_file():
                    return f
    return None

# ----------------- Rotation parsing -----------------
def parse_rotation_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse a rotation or velocity-dispersion table into canonical columns:
      - r_kpc, v_km_s, sigma_km_s, r_m, v_m_s, sigma_m_s
    Attempts to auto-detect radius and velocity columns if names differ.
    """
    df = df.copy()
    radius_names = ["r_kpc", "r", "radius", "radius_kpc", "rad_kpc", "kpc"]
    vel_names = ["v_km_s", "v_obs", "vobs", "v_observed", "v", "v_kms", "vobs_kms", "v_obs_km_s"]
    sig_names = ["sigma_km_s", "sigma", "verr", "err_vobs", "err_v", "sigma_obs"]

    col_r = col_v = col_sig = None
    for c in radius_names:
        if c in df.columns:
            col_r = c
            break
    for c in vel_names:
        if c in df.columns:
            col_v = c
            break
    for c in sig_names:
        if c in df.columns:
            col_sig = c
            break

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if col_r is None:
        for c in numeric_cols:
            med = float(np.nanmedian(df[c].values))
            # pick a plausible radius column in kpc
            if 0.001 < abs(med) < 200.0 and "ra" not in c and "dec" not in c:
                col_r = c
                break
    if col_v is None:
        for c in numeric_cols:
            med = float(np.nanmedian(df[c].values))
            # pick a plausible velocity column in km/s
            if 3.0 < abs(med) < 500.0 and "ra" not in c and "dec" not in c:
                col_v = c
                break

    if col_r is None or col_v is None:
        raise ValueError(f"Unable to find radius/velocity columns. Available columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["r_kpc"] = pd.to_numeric(df[col_r], errors="coerce")
    out["v_km_s"] = pd.to_numeric(df[col_v], errors="coerce")
    if col_sig:
        out["sigma_km_s"] = pd.to_numeric(df[col_sig], errors="coerce")
    else:
        # default observational uncertainty: 5% of velocity or floor 1 km/s
        est = (0.05 * np.abs(out["v_km_s"])).replace(0, np.nan)
        out["sigma_km_s"] = est.fillna(1.0).clip(lower=1.0)

    out["r_m"] = out["r_kpc"] * KPC_TO_M
    out["v_m_s"] = out["v_km_s"] * 1000.0
    out["sigma_m_s"] = out["sigma_km_s"] * 1000.0
    return out.dropna(subset=["r_kpc", "v_km_s"]).reset_index(drop=True)


def load_rotation_for(name: str) -> pd.DataFrame:
    """
    Load and clean a rotation file for 'milky_way', 'ngc3198' or generic name.
    Saves cleaned CSV to OUTPUT_DIR.
    """
    name_key = name.lower()
    if name_key == "milky_way":
        preferred = ["milky_way_rotation_combined.csv", "milky_way_rotation.csv", "milky_way.csv"]
    elif name_key == "ngc3198":
        preferred = ["ngc3198_rotation.csv", "ngc3198.dat", "ngc3198.txt"]
    else:
        preferred = [f"{name_key}_rotation.csv", f"{name_key}.csv"]

    p = _find_file_by_names(preferred)
    if p is None:
        raise FileNotFoundError(f"No rotation file found for '{name}' in search paths: {SEARCH_DIRS}")

    df = _read_table_try(p)
    parsed = parse_rotation_df(df)
    outp = OUTPUT_DIR / f"{name_key}_rotation.csv"
    parsed.to_csv(outp, index=False)
    logger.info("Loaded %s rotation from: %s  (N=%d)", name, str(p), len(parsed))
    logger.info("Saved cleaned %s rotation -> %s", name, str(outp))
    return parsed

# ----------------- Draco building -----------------
def build_draco_sigma_from_vlos(df_stars: pd.DataFrame, n_bins: int = 5, distance_kpc: float = 76.0) -> pd.DataFrame:
    """
    Build binned projected velocity dispersion estimates from a star table
    containing RA/DEC and a radial velocity column (vlos or similar).
    distance_kpc: assumed distance to Draco in kpc (default 76 kpc).
    Returns DataFrame with r_kpc, r_m, sigma_km_s, sigma_m_s, n_stars.
    """
    if df_stars is None or df_stars.shape[0] == 0:
        raise ValueError("Empty Draco star table provided.")
    cols_lower = [c.lower() for c in df_stars.columns]
    # find RA/DEC
    try:
        ra_col = df_stars.columns[cols_lower.index("ra")]
        dec_col = df_stars.columns[cols_lower.index("dec")]
    except ValueError:
        raise ValueError("Draco star table must include 'ra' and 'dec' columns.")
    # find vlos-like column
    vcol = None
    for cand in ("vlos", "v_los", "v_helio", "vhelio", "vrad", "v"):
        if cand in cols_lower:
            vcol = df_stars.columns[cols_lower.index(cand)]
            break
    if vcol is None:
        raise ValueError("Draco star table must include a vlos-like column (vlos/v_helio/vrad).")

    # compute projected radius (small-angle approximation)
    ra0 = float(np.median(df_stars[ra_col].astype(float).values))
    dec0 = float(np.median(df_stars[dec_col].astype(float).values))

    ra_rad = np.deg2rad(df_stars[ra_col].astype(float).values)
    dec_rad = np.deg2rad(df_stars[dec_col].astype(float).values)
    ra0_rad = np.deg2rad(ra0)
    dec0_rad = np.deg2rad(dec0)
    dra = (ra_rad - ra0_rad) * np.cos(dec0_rad)
    ddec = dec_rad - dec0_rad
    theta = np.sqrt(dra * dra + ddec * ddec)  # radians
    R_kpc = theta * distance_kpc
    df2 = pd.DataFrame({"r_kpc_proj": R_kpc, "vlos_kms": pd.to_numeric(df_stars[vcol], errors="coerce")})
    df2 = df2.dropna().sort_values("r_kpc_proj").reset_index(drop=True)
    if df2.shape[0] == 0:
        raise ValueError("No valid Draco vlos entries after parsing.")
    bins = np.array_split(df2, min(n_bins, len(df2)))
    rows = []
    for b in bins:
        if len(b) == 0:
            continue
        r_center = float(np.median(b["r_kpc_proj"].values))
        sigma = float(np.nanstd(b["vlos_kms"].values, ddof=1))
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = 5.0  # fallback floor 5 km/s
        rows.append({
            "r_kpc": r_center,
            "r_m": r_center * KPC_TO_M,
            "sigma_km_s": sigma,
            "sigma_m_s": sigma * 1000.0,
            "n_stars": len(b)
        })
    return pd.DataFrame(rows)


def find_or_build_draco_sigma() -> pd.DataFrame:
    """Try to find a precomputed Draco sigma file or build it from star tables."""
    preferred = ["draco_sigma.csv", "draco_cleaned.csv", "draco_cleaned.txt", "draco.txt", "draco.csv"]
    p = _find_file_by_names(preferred)
    if p is not None:
        df = _read_table_try(p)
        # If it already contains sigma_i columns, try to parse as rotation-like table for consistency
        if "sigma_km_s" in df.columns or "sigma" in df.columns or "sigma_kms" in df.columns:
            try:
                # ensure consistent names
                if "sigma" in df.columns and "sigma_km_s" not in df.columns:
                    df = df.rename(columns={"sigma": "sigma_km_s"})
                parsed = parse_rotation_df(df.rename(columns={"sigma_km_s": "sigma_km_s"}))
                outp = OUTPUT_DIR / "draco_sigma.csv"
                parsed.to_csv(outp, index=False)
                logger.info("Loaded Draco sigma from: %s -> saved %s", str(p), str(outp))
                return parsed
            except Exception:
                pass
        # else try to build from star table
        try:
            sigma_df = build_draco_sigma_from_vlos(df, n_bins=5)
            outp = OUTPUT_DIR / "draco_sigma.csv"
            sigma_df.to_csv(outp, index=False)
            logger.info("Built Draco sigma from: %s  (bins=%d) -> saved %s", str(p), len(sigma_df), str(outp))
            return sigma_df
        except Exception as e:
            raise RuntimeError("Found Draco file but failed to build sigma: " + str(e))
    # fallback: search for any file with 'draco' in name and attempt to build
    for d in SEARCH_DIRS:
        for f in d.glob("*draco*"):
            if f.is_file():
                df = _read_table_try(f)
                try:
                    sigma_df = build_draco_sigma_from_vlos(df, n_bins=5)
                    outp = OUTPUT_DIR / "draco_sigma.csv"
                    sigma_df.to_csv(outp, index=False)
                    logger.info("Built Draco sigma from: %s  (bins=%d) -> saved %s", str(f), len(sigma_df), str(outp))
                    return sigma_df
                except Exception:
                    continue
    raise FileNotFoundError("No Draco file found in search paths.")

# ----------------- Small-scale data -----------------
def load_small_scale_data() -> pd.DataFrame:
    """
    Load tests like Cavendish and Cassini. Expected columns: test, value, error.
    If no file found, create a sensible default with Cavendish and Earth-Sun entries.
    """
    preferred = ["small_scale_tests.csv", "gravity_tests.csv", "eot_wash.csv"]
    p = _find_file_by_names(preferred)
    if p is None:
        # default fallback
        df = pd.DataFrame({
            "test": ["Cavendish", "Earth-Sun"],
            "value": [1.0, 1.0],
            "error": [1e-5, 2e-5]
        })
        outp = OUTPUT_DIR / "small_scale_tests.csv"
        df.to_csv(outp, index=False)
        logger.info("Created default small_scale_tests.csv -> %s", str(outp))
        return df
    df = _read_table_try(p)
    # ensure required columns
    if "test" not in df.columns:
        # try to guess columns
        if df.shape[1] >= 3:
            df = df.rename(columns={df.columns[0]: "test", df.columns[1]: "value", df.columns[2]: "error"})
    outp = OUTPUT_DIR / "small_scale_tests.csv"
    df.to_csv(outp, index=False)
    logger.info("Loaded small-scale data from: %s -> saved %s", str(p), str(outp))
    return df

# ----------------- Model eval & likelihoods -----------------
def evaluate_model_velocity(R_kpc: Sequence[float], model_params: Dict) -> np.ndarray:
    """
    Evaluate circular velocity at radii R_kpc for a given model_params dict.

    Required model_params['bulge'] keys:
      - 'M' : baryonic mass in SOLAR MASSES (float)
      - 'alpha0' : base alpha0 (dimensionless)
      - 'fM' : multiplicative form-factor (dimensionless)

    Returns velocities in km/s (numpy array).
    """
    # Validate inputs
    if "bulge" not in model_params or not isinstance(model_params["bulge"], dict):
        raise KeyError("model_params must contain 'bulge' dict with keys: M, alpha0, fM")
    bulge = model_params["bulge"]
    for k in ("M", "alpha0", "fM"):
        if k not in bulge:
            raise KeyError(f"model_params['bulge'] missing required key '{k}'")

    R_kpc = np.asarray(R_kpc, dtype=float)
    if R_kpc.ndim == 0:
        R_kpc = R_kpc.reshape(1)

    R_m = R_kpc * KPC_TO_M
    # baryonic mass in kg
    M_b_kg = float(bulge["M"]) * M_SUN
    alpha0 = float(bulge["alpha0"])
    fM = float(bulge["fM"])

    # dimensionless radius relative to Yukawa length L
    x_dimless = R_m / L  # R/L

    # mass scaling factor uses 1e42 kg as normalization (manuscript convention)
    mass_factor = M_b_kg / 1e42
    if mass_factor <= 0:
        raise ValueError("Invalid baryonic mass leading to non-positive mass_factor")

    # alpha: alpha0 * (M_b/1e42)^(-GAMMA) ; then multiply by radial cutoff
    alpha_mass = alpha0 * (mass_factor) ** (-GAMMA)
    # interpret X0 as dimensionless threshold on R/L. cutoff = min(1, (R/L / X0)^2)
    cutoff = np.minimum(1.0, (x_dimless / X0) ** 2)
    alpha = alpha_mass * cutoff

    M_5 = alpha * M_b_kg * fM  # effective 5D mass in kg

    # Yukawa-like correction factor (uses R_m/L)
    yukawa_factor = np.exp(-R_m / L) * (1.0 + (R_m / L))

    # ensure no division by zero
    R_m_safe = np.where(R_m > 0.0, R_m, np.finfo(float).tiny)

    v2 = (G_4 * M_b_kg / R_m_safe) + (G_4 * M_5 / R_m_safe) * yukawa_factor

    # numerical safety: force non-negative
    v2 = np.where(v2 > 0.0, v2, 0.0)
    v_m_s = np.sqrt(v2)
    return (v_m_s * 1e-3).reshape(R_kpc.shape)  # return km/s

def log_likelihood_rotation(model_params: Dict, rot_df: pd.DataFrame, jitter_km_s: float = 0.0) -> float:
    """
    Gaussian log-likelihood for rotation curve points.
    rot_df must contain columns 'r_kpc', 'v_km_s', 'sigma_km_s'.
    jitter_km_s: additional error term added in quadrature.
    """
    R = rot_df["r_kpc"].values
    Vobs = rot_df["v_km_s"].values
    Verr = rot_df.get("sigma_km_s", np.full_like(Vobs, np.nan))
    Verr = np.where(np.isnan(Verr) | (Verr <= 0.0), (0.05 * np.abs(Vobs) + 1.0), Verr)
    sigma2 = Verr ** 2 + float(jitter_km_s) ** 2
    Vmodel = evaluate_model_velocity(R, model_params)
    resid2 = (Vobs - Vmodel) ** 2
    lnL = -0.5 * np.sum(resid2 / sigma2 + np.log(2.0 * np.pi * sigma2))
    if not np.isfinite(lnL):
        logger.warning("Non-finite lnL in log_likelihood_rotation; returning large negative")
        return float("-1e30")
    return float(lnL)


def log_likelihood_draco_dispersion(model_params: Dict, draco_sigma_df: pd.DataFrame) -> float:
    """
    Likelihood for Draco dispersion bins. Expects draco_sigma_df with r_kpc, sigma_km_s.
    We evaluate the model at those radii and compare predicted sigma = v_circ / sqrt(3).
    This function does not mutate the user's model_params.
    """
    # copy model_params and override bulge mass to Draco baryonic mass (6e35 kg converted to solar masses)
    mp = {k: (v.copy() if isinstance(v, dict) else v) for k, v in model_params.items()}
    mp["bulge"] = mp.get("bulge", {}).copy()
    mp["bulge"]["M"] = 6e35 / M_SUN  # convert kg -> solar masses

    R = draco_sigma_df["r_kpc"].values
    sigma_obs = draco_sigma_df["sigma_km_s"].values
    sigma_err = np.maximum(0.1 * sigma_obs, 1.0)  # 10% or floor 1 km/s
    Vc = evaluate_model_velocity(R, mp)  # km/s
    sigma_pred = Vc / np.sqrt(3.0)
    resid2 = (sigma_obs - sigma_pred) ** 2
    sigma2 = sigma_err ** 2
    lnL = -0.5 * np.sum(resid2 / sigma2 + np.log(2.0 * np.pi * sigma2))
    if not np.isfinite(lnL):
        logger.warning("Non-finite lnL in log_likelihood_draco_dispersion; returning large negative")
        return float("-1e30")
    return float(lnL)


def log_likelihood_small_scale(model_params: Dict, small_scale_df: pd.DataFrame) -> float:
    """
    Small-scale tests likelihood (Cavendish & Earth-Sun/Cassini).
    small_scale_df must contain columns ['test', 'value', 'error'] (test names recognized case-insensitively).
    Matching is robust to row order.
    """
    if "bulge" not in model_params or "alpha0" not in model_params["bulge"]:
        raise KeyError("model_params['bulge']['alpha0'] required for small-scale likelihood")

    alpha0 = float(model_params["bulge"]["alpha0"])

    # Cavendish: use local test geometry M=1 kg, r=0.1 m -> dimensionless r/L
    M_cav_kg = 1.0
    r_cav_m = 0.1
    x_cav_dimless = r_cav_m / L
    alpha_cav = alpha0 * (M_cav_kg / 1e42) ** (-GAMMA) * np.minimum(1.0, (x_cav_dimless / X0) ** 2)
    G_eff_G4_pred = 1.0 + float(alpha_cav)

    # Earth-Sun: use solar mass M_SUN and 1 AU
    M_sun_kg = M_SUN
    r_au_m = 1.496e11
    x_au_dimless = r_au_m / L
    alpha_au = alpha0 * (M_sun_kg / 1e42) ** (-GAMMA) * np.minimum(1.0, (x_au_dimless / X0) ** 2)
    gamma_pred = 1.0 + float(alpha_au)

    # match predictions to rows by test name
    tests = small_scale_df["test"].astype(str).str.lower().values
    preds = np.full(len(tests), np.nan, dtype=float)
    for i, t in enumerate(tests):
        if "cavendish" in t or "cav" in t:
            preds[i] = G_eff_G4_pred
        elif "earth" in t or "sun" in t or "cassini" in t:
            preds[i] = gamma_pred
        else:
            # unknown: leave NaN (we'll ignore these rows)
            preds[i] = np.nan

    mask = np.isfinite(preds)
    if not np.any(mask):
        logger.warning("No matching small-scale tests found in DataFrame; returning large negative lnL")
        return float("-1e30")

    values = small_scale_df["value"].values.astype(float)
    errors = small_scale_df["error"].values.astype(float)
    resid2 = (preds[mask] - values[mask]) ** 2
    sigma2 = errors[mask] ** 2
    lnL = -0.5 * np.sum(resid2 / sigma2 + np.log(2.0 * np.pi * sigma2))
    if not np.isfinite(lnL):
        logger.warning("Non-finite lnL in log_likelihood_small_scale; returning large negative")
        return float("-1e30")
    return float(lnL)

# ----------------- Convenience demo loader -----------------
def try_load_all() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Attempt to load Milky Way, NGC3198, Draco sigma and small-scale CSVs.
    Returns: (mw_df, ngc_df, draco_df_or_None, small_scale_df_or_None)
    """
    mw = ngc = draco = small_scale = None
    try:
        mw = load_rotation_for("milky_way")
    except Exception as e:
        logger.info("Milky Way rotation not found: %s", e)
    try:
        ngc = load_rotation_for("ngc3198")
    except Exception as e:
        logger.info("NGC3198 rotation not found: %s", e)
    try:
        draco = find_or_build_draco_sigma()
    except Exception as e:
        logger.info("Draco sigma not found / could not be built: %s", e)
    try:
        small_scale = load_small_scale_data()
    except Exception as e:
        logger.info("Small-scale data not found: %s", e)
    return mw, ngc, draco, small_scale