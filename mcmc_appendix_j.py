#!/usr/bin/env python3
# mcmc_quasar_full.py
"""
Full, robust MCMC runner for Appendix F (DESI quasar P(k) fits).

Features
- robust ClassCWD import (tries common module locations)
- safe emcee HDF5 backend handling (removes corrupt file and retries; falls back to .npy)
- optional multiprocessing Pool (falls back to single-process on restricted platforms)
- saves: mcmc_chain.npy, mcmc_flat_samples.npy, trace_plot.png, corner_plot.png
- quick mode (reduced walkers/steps) via --quick CLI arg or QUICK_RUN=True
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

# --- Try to import local ClassCWD wrapper from a few likely locations ----------
ClassCWD = None
_class_import_errors = []
_try_modules = [
    "class_cwd",  # same folder: class_cwd.py
    "appendix_f.class_cwd.class_cwd",  # nested module path
    "appendix_f.class_cwd",  # package-like
    "class_cwd.class_cwd",
]
for mod in _try_modules:
    try:
        components = mod.split(".")
        # dynamic import
        module = __import__(mod, fromlist=["ClassCWD"])
        # prefer ClassCWD attribute
        if hasattr(module, "ClassCWD"):
            ClassCWD = getattr(module, "ClassCWD")
            break
        # maybe module itself is the class
        if isinstance(module, type) and module.__name__ == "ClassCWD":
            ClassCWD = module
            break
    except Exception as e:
        _class_import_errors.append((mod, repr(e)))

if ClassCWD is None:
    # final attempt: try importing appendix_f.class_cwd via file sys.path trick
    try:
        # if appendix_f/class_cwd/class_cwd.py exists, add its parent to path:
        possible = Path.cwd() / "appendix_f" / "class_cwd" / "class_cwd.py"
        if possible.exists():
            sys.path.insert(0, str(possible.parent))
            from class_cwd import ClassCWD  # type: ignore
    except Exception as e:
        _class_import_errors.append(("appendix_f/class_cwd/class_cwd.py (file-try)", repr(e)))

if ClassCWD is None:
    err_msg = (
        "Could not import ClassCWD. Please place class_cwd.py in one of:\n"
        " - same folder as this script (class_cwd.py)\n"
        " - appendix_f/class_cwd/class_cwd.py\n"
        " - or install a package that provides ClassCWD.\n"
        "Tried these import attempts:\n"
    )
    for m, e in _class_import_errors:
        err_msg += f"  {m!s}: {e}\n"
    # do not hard-exit here; we will try to run but will raise when needed
    print(err_msg, file=sys.stderr)

# --- emcee & corner imports (required) -------------------------------------
try:
    import emcee
except Exception as e:
    raise ImportError("emcee is required. Install with: pip install emcee") from e

try:
    import corner  # optional but nice
except Exception:
    corner = None

# --- Configuration: edit paths/settings as needed ----------------------------
# Default data directory (adjust for your device); keep safe defaults
DATA_DIR = Path("/storage/emulated/0/Download")
if not DATA_DIR.exists():
    DATA_DIR = Path.cwd()

OUT_DIR = DATA_DIR / "cwd_quasar_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DESI_PK_TXT = DATA_DIR / "desi_quasar_pk_z3.txt"  # expected columns: k, P_obs, diag_var
DESI_COV_NPY = DATA_DIR / "desi_quasar_cov.npy"  # optional full covariance

# Default MCMC settings (overridden in quick mode)
DEFAULTS = {
    "N_WALKERS": 48,
    "N_STEPS": 3000,
    "BURN_IN": 800,
    "RANDOM_SEED": 1234,
}

# Model defaults
Z = 3.0
FIX_BETA = 0.30  # Kaiser beta used in stub
USE_HDF_BACKEND = True  # attempt to use emcee HDFBackend (will auto-fallback)

# Parameterization:
# theta = [alpha0, log10_L_mpc, gamma, b, sigma_v_kms, ln_jitter]
NDIM = 6

# -------------------------
# Utility functions
# -------------------------
def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load DESI P(k) text file. Expect at least 3 columns: k, P_obs, diag_var.
    If desi_quasar_cov.npy exists, use it as full covariance.
    Returns (k, P_obs, cov_matrix)
    """
    if not DESI_PK_TXT.exists():
        raise FileNotFoundError(f"DESI pk file not found: {DESI_PK_TXT}")

    data = np.loadtxt(DESI_PK_TXT)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("DESI pk file must have columns: k, P_obs, diag_var")

    k = data[:, 0].astype(float)
    P_obs = data[:, 1].astype(float)
    diag_var = data[:, 2].astype(float)

    if DESI_COV_NPY.exists():
        cov = np.load(DESI_COV_NPY)
        if cov.shape[0] != len(k) or cov.shape[1] != len(k):
            print("Warning: covariance shape mismatch; falling back to diagonal variances.")
            cov = np.diag(diag_var)
    else:
        cov = np.diag(diag_var)
    return k, P_obs, cov


def log_prior(theta: np.ndarray) -> float:
    alpha0, log10_L_mpc, gamma, b, sigma_v_kms, ln_jitter = theta
    if not (0.0 <= alpha0 <= 5.0):
        return -np.inf
    if not (-1.0 <= log10_L_mpc <= 2.5):
        return -np.inf
    if not (0.0 <= gamma <= 2.0):
        return -np.inf
    if not (1.0 <= b <= 6.0):
        return -np.inf
    if not (20.0 <= sigma_v_kms <= 800.0):
        return -np.inf
    if not (-20.0 <= ln_jitter <= 10.0):
        return -np.inf
    # flat priors; could add weak Gaussian prior on alpha0 if desired
    return 0.0


def build_params_from_theta(theta: np.ndarray):
    alpha0, log10_L_mpc, gamma, b, sigma_v_kms, ln_jitter = theta
    params = {
        "alpha0": float(alpha0),
        "L_mpc": float(10.0 ** float(log10_L_mpc)),
        "gamma": float(gamma),
    }
    return params, float(b), float(sigma_v_kms), float(ln_jitter)


def safe_get_cwd_wrapper(params: dict):
    """
    Lazy singleton to create/reuse a ClassCWD wrapper instance.
    If ClassCWD is not importable, raise an informative error.
    """
    if ClassCWD is None:
        raise ImportError("ClassCWD is not available. Place class_cwd.py in your project (see earlier message).")
    # reuse an instance if present
    if not hasattr(safe_get_cwd_wrapper, "instance"):
        try:
            # instantiate in 'auto' backend mode so wrapper decides whether CLASS is present
            safe_get_cwd_wrapper.instance = ClassCWD({"cwd": params}, backend="auto")
        except Exception as e:
            # try fallback: instantiate using minimal dict
            safe_get_cwd_wrapper.instance = ClassCWD({"cwd": params})
    else:
        # update params if wrapper provides a way (we call get_pquasar with overrides later)
        pass
    return safe_get_cwd_wrapper.instance


def safe_log_like(theta: np.ndarray, k: np.ndarray, P_obs: np.ndarray, cov: np.ndarray) -> float:
    """
    Compute Gaussian log-likelihood robustly, adding a jitter term.
    Returns lnL (float).
    """
    params, b, sigma_v_kms, ln_jitter = build_params_from_theta(theta)
    jitter_var = np.exp(ln_jitter) ** 2

    # create/reuse wrapper instance
    try:
        cwd = safe_get_cwd_wrapper({"alpha0": params["alpha0"], "L_mpc": params["L_mpc"], "gamma": params["gamma"]})
    except Exception as e:
        # If wrapper can't be instantiated, strongly penalize
        print("ERROR: Could not initialize ClassCWD wrapper:", e, file=sys.stderr)
        return -1e99

    # Get model prediction. ClassCWD.get_pquasar signature in stub:
    # get_pquasar(k_hmpc, z, b=3.5, beta=0.3, sigma_v_kms=200.0, alpha0=None, L_mpc=None, gamma=None)
    try:
        P_model = cwd.get_pquasar(k, z=Z, b=b, beta=FIX_BETA, sigma_v_kms=sigma_v_kms,
                                  alpha0=params["alpha0"], L_mpc=params["L_mpc"], gamma=params["gamma"])
    except TypeError:
        # fallback in case signature differs slightly
        P_model = cwd.get_pquasar(k, z=Z, b=b, beta=FIX_BETA, sigma_v_kms=sigma_v_kms)

    d = P_obs - P_model

    # add jitter on diagonal
    C = cov.copy()
    if jitter_var > 0.0:
        C = C + np.eye(C.shape[0]) * jitter_var

    # compute robust log-likelihood
    try:
        sign, logdet = np.linalg.slogdet(C)
        if sign <= 0:
            # numerical breakdown -> strongly penalize
            return -1e99
        invC = np.linalg.inv(C)
        chi2 = float(d @ (invC @ d))
        lnL = -0.5 * (chi2 + logdet + len(d) * np.log(2.0 * np.pi))
        return float(lnL)
    except np.linalg.LinAlgError:
        return -1e99


def log_probability(theta: np.ndarray, k: np.ndarray, P_obs: np.ndarray, cov: np.ndarray) -> float:
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = safe_log_like(theta, k, P_obs, cov)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# -------------------------
# Robust MCMC runner
# -------------------------
def run_mcmc(
    nwalkers: int,
    nsteps: int,
    burn_in: int,
    random_seed: int,
    use_hdf_backend: bool = True,
    quick_mode: bool = False,
):
    rng = np.random.default_rng(random_seed)

    # load data
    k, P_obs, cov = load_data()
    N = len(k)
    print(f"Loaded data from: {DESI_PK_TXT}")
    print(f"  N = {N}, k-range = [{k.min():.3g}, {k.max():.3g}]")
    print("Outputs ->", OUT_DIR)

    # initial walker positions (center of priors)
    p0_center = np.array([1.05, np.log10(10.0), 0.48, 3.5, 200.0, np.log(1e-6)])
    p0 = p0_center + 1e-3 * rng.standard_normal(size=(nwalkers, NDIM))
    # ensure walkers inside priors
    for i in range(nwalkers):
        tries = 0
        while not np.isfinite(log_prior(p0[i])) and tries < 200:
            p0[i] = p0_center + 1e-2 * rng.standard_normal(size=NDIM)
            tries += 1
        if not np.isfinite(log_prior(p0[i])):
            raise RuntimeError("Failed to initialize all walkers inside priors; try fewer walkers or wider priors.")

    # Setup HDF backend robustly
    backend = None
    backend_file = OUT_DIR / "mcmc_quasar_chain.h5"
    use_backend = False
    if use_hdf_backend:
        try:
            backend = emcee.backends.HDFBackend(str(backend_file))
            try:
                backend.reset(nwalkers, NDIM)
            except Exception as e_reset:
                # try removing a possibly corrupted file and re-create
                try:
                    if backend_file.exists():
                        print("Warning: resetting backend failed; removing file and retrying...", file=sys.stderr)
                        backend_file.unlink(missing_ok=True)
                    backend = emcee.backends.HDFBackend(str(backend_file))
                    backend.reset(nwalkers, NDIM)
                except Exception as e_retry:
                    print("HDF backend creation/reset retry failed; will run without HDF backend.", file=sys.stderr)
                    backend = None
            if backend is not None:
                use_backend = True
        except Exception as e:
            print("HDF backend unavailable (h5py/hdf error). Running without HDF backend.", file=sys.stderr)
            backend = None
            use_backend = False
    else:
        print("HDF backend disabled by configuration.")

    # Setup multiprocessing pool safely
    pool = None
    nproc = 1
    try:
        cpu_count = mp.cpu_count()
        nproc = max(1, min(cpu_count, nwalkers))
        if nproc > 1:
            # use spawn on some platforms to be safer; fall back silently if it fails
            try:
                ctx = mp.get_context("spawn")
            except Exception:
                ctx = mp
            pool = ctx.Pool(processes=nproc)
    except Exception as e:
        print("Could not create multiprocessing pool; proceeding single-threaded.", file=sys.stderr)
        pool = None
        nproc = 1

    print(f"Starting emcee: walkers={nwalkers}, steps={nsteps}, burn_in={burn_in}, nproc={nproc}, HDF_backend={use_backend}")

    sampler = emcee.EnsembleSampler(nwalkers, NDIM, log_probability, args=(k, P_obs, cov), backend=backend, pool=pool)

    t0 = time.time()
    try:
        try:
            sampler.run_mcmc(p0, nsteps, progress=True)
        except TypeError:
            sampler.run_mcmc(p0, nsteps)
    except KeyboardInterrupt:
        print("Interrupted by user; finishing early...", file=sys.stderr)
    finally:
        # close pool if used
        if pool is not None:
            try:
                pool.close()
                pool.join()
            except Exception:
                pass

    elapsed = time.time() - t0
    acc = np.nan
    try:
        acc = float(np.mean(sampler.acceptance_fraction))
    except Exception:
        pass
    print(f"MCMC finished in {elapsed:.1f}s; mean acceptance_fraction = {acc:.3f}")

    # Extract chains
    try:
        chain = sampler.get_chain()
    except Exception:
        chain = None
    try:
        flat = sampler.get_chain(discard=burn_in, flat=True)
    except Exception:
        # older emcee versions use get_chain(flat=True, discard=burn_in)
        try:
            flat = sampler.get_chain(flat=True, discard=burn_in)
        except Exception:
            flat = None

    # Save .npy outputs always
    if chain is not None:
        np.save(OUT_DIR / "mcmc_chain.npy", chain)
        print("Saved full chain ->", OUT_DIR / "mcmc_chain.npy")
    else:
        print("No chain saved (chain is None).")

    if flat is not None:
        np.save(OUT_DIR / "mcmc_flat_samples.npy", flat)
        print("Saved flattened samples ->", OUT_DIR / "mcmc_flat_samples.npy")
    else:
        print("No flattened samples available.")

    if use_backend and backend_file.exists():
        print("HDF backend file:", backend_file)
    else:
        print("HDF backend not used (or failed); outputs are in .npy files in", OUT_DIR)

    # Trace plots
    labels = ["alpha0", "log10_L_mpc", "gamma", "b", "sigma_v_kms", "ln_jitter"]
    if chain is not None:
        fig, axes = plt.subplots(NDIM, 1, figsize=(8, 2.0 * NDIM), sharex=True)
        steps = chain.shape[1]
        xaxis = np.arange(steps)
        for i in range(NDIM):
            axes[i].plot(xaxis, chain[:, :, i].T, color="C0", alpha=0.5)
            axes[i].set_ylabel(labels[i])
        axes[-1].set_xlabel("step")
        fig.tight_layout()
        trace_path = OUT_DIR / "trace_plot.png"
        fig.savefig(trace_path, dpi=150)
        plt.close(fig)
        print("Saved trace plot ->", trace_path)

    # Corner plot
    if flat is not None and corner is not None:
        try:
            fig = corner.corner(flat, labels=labels, show_titles=True, quantiles=[0.16, 0.5, 0.84])
            corner_path = OUT_DIR / "corner_plot.png"
            fig.savefig(corner_path, dpi=150)
            plt.close(fig)
            print("Saved corner plot ->", corner_path)
        except Exception as e:
            print("Corner plot generation failed:", e, file=sys.stderr)
    elif flat is not None:
        print("corner not installed; install with `pip install corner` to get corner plot.")

    # Summary of parameter posteriors
    if flat is not None:
        pct = np.percentile(flat, [16, 50, 84], axis=0)
        print("Parameter estimates (16/50/84 percentiles):")
        for i, lab in enumerate(labels):
            p16, p50, p84 = pct[0, i], pct[1, i], pct[2, i]
            print(f"  {lab:12s} = {p50:.4g} (+{p84-p50:.4g}/-{p50-p16:.4g})")

    return sampler, flat


# -------------------------
# CLI & entrypoint
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MCMC runner for Appendix F (DESI quasar P(k))")
    p.add_argument("--quick", action="store_true", help="Quick mode: fewer walkers/steps (useful on mobile)")
    p.add_argument("--no-hdf", action="store_true", help="Disable HDF backend (force .npy saving)")
    p.add_argument("--seed", type=int, default=DEFAULTS["RANDOM_SEED"], help="Random seed")
    p.add_argument("--nwalkers", type=int, default=DEFAULTS["N_WALKERS"], help="Number of walkers")
    p.add_argument("--nsteps", type=int, default=DEFAULTS["N_STEPS"], help="Number of MCMC steps")
    p.add_argument("--burn", type=int, default=DEFAULTS["BURN_IN"], help="Burn-in to discard")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    QUICK_RUN = args.quick
    if QUICK_RUN:
        print("Quick mode enabled: reducing walkers/steps for faster run.")
        N_WALKERS = max(4, min(24, args.nwalkers // 4))
        N_STEPS = max(100, args.nsteps // 10)
        BURN_IN = max(10, args.burn // 10)
    else:
        N_WALKERS = args.nwalkers
        N_STEPS = args.nsteps
        BURN_IN = args.burn

    USE_HDF = not args.no_hdf and USE_HDF_BACKEND

    print("Starting MCMC runner (mcmc_quasar_full.py)")
    print("Data dir:", DATA_DIR)
    print("Output dir:", OUT_DIR)
    print(f"N_walkers={N_WALKERS}, N_steps={N_STEPS}, burn_in={BURN_IN}, quick={QUICK_RUN}, use_hdf={USE_HDF}")

    sampler, flat = run_mcmc(N_WALKERS, N_STEPS, BURN_IN, args.seed, use_hdf_backend=USE_HDF, quick_mode=QUICK_RUN)
    print("Finished. Look for outputs in:", OUT_DIR)