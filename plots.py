#!/usr/bin/env python3
# plots.py
"""
Create publication-ready plots for:
 - Milky Way rotation curve (data + best-fit model + optional Yukawa correction)
 - NGC3198 rotation curve (same)
 - Draco dispersion profile (data binned vs model)

Assumes these modules/files exist (from prior steps):
 - cw_model.py with build_model(...) and CompositeModel.vc_total(...)
 - likelihood.py with find_or_load_rotation(...) and find_or_build_draco_sigma(...)
 - outputs/mcmc_flat_samples.npy  OR outputs/mcmc_chain.npy
 - data/milky_way_rotation*.csv, data/ngc3198_rotation.csv, data/draco_sigma.csv

Saves plots into OUTPUT_DIR (tries a few sensible locations).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# try to import local helpers
try:
    from likelihood import try_load_all
except Exception as e:
    raise ImportError("Could not import helpers from likelihood.py. Make sure likelihood.py is in PYTHONPATH.") from e

try:
    from cw_model import build_model
except Exception as e:
    raise ImportError("Could not import build_model from cw_model.py. Make sure cw_model.py is in PYTHONPATH.") from e

# Preferred output directory (match previous runs)
DEFAULT_OUTPUTS = [
    Path("/storage/emulated/0/Download/cwd_outputs"),
    Path.cwd() / "outputs",
    Path("/storage/emulated/0/Download"),
    Path("/storage/emulated/0")
]
OUTPUT_DIR = next((p for p in DEFAULT_OUTPUTS if p.exists()), DEFAULT_OUTPUTS[0])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Plots will be written to:", OUTPUT_DIR)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_bestfit_params_from_mcmc(outputs_dir: Path) -> dict:
    """
    Look for mcmc_flat_samples.npy or mcmc_chain.npy and return median params dict
    using the parameter order:
      [log10_M200, c200, log10_Mdisk, Rd, log10_Mbulge, a, jitter]
    """
    flat_path = outputs_dir / "mcmc_flat_samples.npy"
    chain_path = outputs_dir / "mcmc_flat_samples.npy"  # same name as earlier code
    alt_chain = outputs_dir / "mcmc_chain.npy"
    if flat_path.exists():
        arr = np.load(flat_path)
    elif alt_chain.exists():
        arr = np.load(alt_chain)
        # try to find flattened shape: if shape is (nwalkers, nsteps, ndim), flatten last two dims
        if arr.ndim == 3:
            arr = arr.reshape((-1, arr.shape[-1]))
    else:
        raise FileNotFoundError("No mcmc_flat_samples.npy or mcmc_chain.npy found in outputs; run MCMC first.")
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError(f"Unexpected MCMC array shape: {arr.shape}")
    median = np.median(arr, axis=0)
    keys = ["log10_M200", "c200", "log10_Mdisk", "Rd", "log10_Mbulge", "a", "jitter"]
    param_dict = dict(zip(keys, median[: len(keys)]))
    # convert to build_model format
    params = {
        "halo": {"type": "nfw", "M200": 10 ** (float(param_dict["log10_M200"])), "c200": float(param_dict["c200"])},
        "disk": {"M": 10 ** (float(param_dict["log10_Mdisk"])), "Rd": float(param_dict["Rd"])},
        "bulge": {"M": 10 ** (float(param_dict["log10_Mbulge"])), "a": float(param_dict["a"])},
    }
    return params, param_dict  # return both for printing


def yukawa_corrected_velocity(V_model: np.ndarray, r_kpc: np.ndarray, alpha: float = 0.1, lambda_kpc: float = 50.0) -> np.ndarray:
    """
    Apply a phenomenological Yukawa-like multiplicative correction:
      V_yuk(r) = V_model(r) * sqrt(1 + alpha * exp(-r / lambda_kpc))
    This is a simple approximation — replace with the exact formula from Appendix D if required.
    """
    factor = 1.0 + alpha * np.exp(-np.asarray(r_kpc) / float(lambda_kpc))
    return V_model * np.sqrt(np.maximum(factor, 0.0))


def chi2(Vobs, Verr, Vmodel):
    """Return chi2 and reduced-chi2 (reduced=chi2/ndof) where ndof = N-1."""
    mask = np.isfinite(Vobs) & np.isfinite(Verr) & np.isfinite(Vmodel)
    if mask.sum() == 0:
        return np.nan, np.nan
    resid = (Vobs[mask] - Vmodel[mask]) ** 2
    var = Verr[mask] ** 2
    # avoid zeros
    var = np.where(var <= 0, (0.05 * np.abs(Vobs[mask]) + 1.0) ** 2, var)
    chi2val = np.sum(resid / var)
    ndof = mask.sum() - 1
    return float(chi2val), float(chi2val / ndof) if ndof > 0 else np.nan

# ---------------------------------------------------------------------
# Plotting routines
# ---------------------------------------------------------------------
def plot_rotation_curve(rot_df: pd.DataFrame, params: dict, param_summary: dict, title: str, fname: str,
                        alpha_yuk=0.12, lambda_kpc=50.0, rmax_plot=200.0):
    """
    rot_df: standardized rotation dataframe with columns r_kpc, v_km_s, sigma_km_s
    params: dict accepted by build_model(...)
    param_summary: dict of raw median parameters for printing (optional)
    fname: filename to save
    """
    # prepare R grid
    r_grid = np.linspace(0.1, rmax_plot, 500)
    # evaluate model
    model = build_model(params)
    try:
        v_model_grid = np.asarray(model.vc_total(r_grid), dtype=float)
    except Exception:
        comps = model.vc_components(r_grid)
        v2 = np.zeros_like(r_grid)
        for v in comps.values():
            v2 += np.asarray(v) ** 2
        v_model_grid = np.sqrt(v2)

    # best-fit at data radii
    r_data = rot_df["r_kpc"].values
    v_data = rot_df["v_km_s"].values
    verr = rot_df["sigma_km_s"].values

    try:
        v_model_at_data = np.asarray(model.vc_total(r_data), dtype=float)
    except Exception:
        comps = model.vc_components(r_data)
        v2d = np.zeros_like(r_data)
        for v in comps.values():
            v2d += np.asarray(v) ** 2
        v_model_at_data = np.sqrt(v2d)

    # Yukawa curve
    v_yuk_grid = yukawa_corrected_velocity(v_model_grid, r_grid, alpha=alpha_yuk, lambda_kpc=lambda_kpc)

    # Chi2
    chi2val, red = chi2(v_data, verr, v_model_at_data)

    # plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(r_data, v_data, yerr=verr, fmt='o', markersize=6, alpha=0.8, label="Data")
    ax.plot(r_grid, v_model_grid, color="red", lw=2, label="Best-fit model")
    ax.plot(r_grid, v_yuk_grid, color="green", ls="--", lw=2, label=f"Yukawa (α={alpha_yuk}, λ={lambda_kpc} kpc)")
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel("V [km/s]")
    ax.set_title(title)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.2)

    # text with param summary + chi2
    txt = []
    if param_summary is not None:
        txt.append("Best-fit (median):")
        for k, v in param_summary.items():
            txt.append(f"{k}={v:.3g}")
    txt.append(f"χ² = {chi2val:.2f}, χ²_red = {red:.3f}")
    ax.text(0.98, 0.02, "\n".join(txt), transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=6))

    outpath = OUTPUT_DIR / fname
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print("Saved:", outpath)


def plot_draco_dispersion(draco_df: pd.DataFrame, params: dict, param_summary: dict, fname: str, alpha_yuk=0.12, lambda_kpc=50.0):
    """
    draco_df expected columns: r_kpc, sigma_km_s, sigma_m_s (optional)
    model mapping: sigma_pred = vc(r)/sqrt(3) (simple isotropic mapping)
    """
    r_grid = np.linspace(0.01, np.max(np.concatenate([draco_df["r_kpc"].values, [10.0]])), 300)
    model = build_model(params)
    try:
        v_model_grid = np.asarray(model.vc_total(r_grid), dtype=float)
    except Exception:
        comps = model.vc_components(r_grid)
        v2 = np.zeros_like(r_grid)
        for v in comps.values():
            v2 += np.asarray(v) ** 2
        v_model_grid = np.sqrt(v2)

    sigma_pred = v_model_grid / np.sqrt(3.0)
    sigma_pred_yuk = yukawa_corrected_velocity(v_model_grid, r_grid, alpha=alpha_yuk, lambda_kpc=lambda_kpc) / np.sqrt(3.0)

    # data
    r_data = draco_df["r_kpc"].values
    sigma_obs = draco_df["sigma_km_s"].values
    nstars = draco_df.get("n_stars", pd.Series([None]*len(r_data))).values

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(r_data, sigma_obs, yerr=0.1 * sigma_obs + 1.0, fmt='o', label="Draco dispersion (binned)")
    ax.plot(r_grid, sigma_pred, label="Model (isotropic mapping)")
    ax.plot(r_grid, sigma_pred_yuk, '--', label=f"Model + Yukawa (α={alpha_yuk})")
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel(r"$\sigma$ [km/s]")
    ax.set_title("Draco velocity dispersion (binned)")
    ax.legend()
    ax.grid(alpha=0.2)
    outpath = OUTPUT_DIR / fname
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print("Saved:", outpath)

# ---------------------------------------------------------------------
# Main : glue everything together
# ---------------------------------------------------------------------
def main():
    # 🔹 Load all rotation + Draco data in one call
    try:
        rot_mw, rot_ngc, draco_sigma = try_load_all()
        print("Data successfully loaded via try_load_all().")
    except Exception as e:
        print("ERROR: Could not load data with try_load_all():", e)
        rot_mw, rot_ngc, draco_sigma = None, None, None

    # 🔹 Load best-fit params from MCMC
    try:
        params, summary = load_bestfit_params_from_mcmc(OUTPUT_DIR)
        print("Loaded best-fit params (median) from MCMC.")
    except Exception as e:
        print("WARNING: Could not load MCMC results:", e)
        print("Using a default example parameter set.")
        params = {
            "halo": {"type": "nfw", "M200": 1.2e12, "c200": 10.0},
            "disk": {"M": 6.0e10, "Rd": 3.0},
            "bulge": {"M": 8.0e9, "a": 0.6},
        }
        summary = {
            "log10_M200": 12.3, "c200": 10.0,
            "log10_Mdisk": 10.0, "Rd": 3.0,
            "log10_Mbulge": 9.0, "a": 0.6,
            "jitter": 5.0
        }

    # 🔹 Yukawa defaults
    alpha_yuk = 0.12      # dimensionless
    lambda_kpc = 50.0     # kpc

    # ---------- Milky Way ----------
    if rot_mw is not None:
        print(f"Loaded milky_way rotation (N={len(rot_mw)})")
        plot_rotation_curve(rot_mw, params, summary,
                            title="Milky Way Rotation Curve",
                            fname="milky_way_rotation_plot.png",
                            alpha_yuk=alpha_yuk, lambda_kpc=lambda_kpc,
                            rmax_plot=200.0)
    else:
        print("Milky Way rotation not available.")

    # ---------- NGC3198 ----------
    if rot_ngc is not None:
        print(f"Loaded ngc3198 rotation (N={len(rot_ngc)})")
        plot_rotation_curve(rot_ngc, params, summary,
                            title="NGC3198 Rotation Curve",
                            fname="ngc3198_rotation_plot.png",
                            alpha_yuk=alpha_yuk, lambda_kpc=lambda_kpc,
                            rmax_plot=50.0)
    else:
        print("NGC3198 rotation not available.")

    # ---------- Draco ----------
    if draco_sigma is not None:
        print(f"Loaded draco_sigma (Nbins={len(draco_sigma)})")
        plot_draco_dispersion(draco_sigma, params, summary,
                              fname="draco_dispersion_plot.png",
                              alpha_yuk=alpha_yuk, lambda_kpc=lambda_kpc)
    else:
        print("Draco sigma not available.")

    print("✅ All plotting done. Files are in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()