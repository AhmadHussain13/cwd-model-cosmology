# mcmc_runner.py
"""
Safer MCMC runner that uses likelihood.py helpers.

- Writes outputs to /storage/emulated/0/Download/cwd_outputs
- Falls back if emcee HDF5 backend fails (saves numpy arrays)
- Logs exceptions instead of silently returning -inf
"""

import sys
import time
import logging
from pathlib import Path
import importlib.util
from typing import Tuple, Optional

import numpy as np

# Optional plotting / corner dependencies; handle gracefully
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import corner
except Exception:
    corner = None

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cwd.mcmc_runner")

# ---------------- Output / search paths ----------------
OUTPUT_DIR = Path("/storage/emulated/0/Download/cwd_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_DIRS = [
    Path("/storage/emulated/0/Download"),
    Path("/storage/emulated/0/data"),
    Path("/storage/emulated/0"),
    Path.cwd(),
]

# ---------------- Robust likelihood.py loader ----------------
def _load_likelihood_module() -> object:
    """
    Load likelihood.py either via normal import or by searching common storage locations.
    Returns loaded module object. Raises ModuleNotFoundError / ImportError on failure.
    """
    # try standard import first
    try:
        import likelihood as _lik  # type: ignore
        logger.info("Imported 'likelihood' via normal import.")
        return _lik
    except Exception:
        logger.debug("Standard import of 'likelihood' failed; searching paths.")

    candidates = [
        Path.cwd(),
        Path(__file__).parent if "__file__" in globals() else Path.cwd(),
        Path("/storage/emulated/0/Download"),
        Path("/storage/emulated/0/Download/cwd_outputs"),
        Path("/storage/emulated/0"),
        Path("/sdcard"),
    ]

    found = None
    for d in candidates:
        try:
            p = Path(d) / "likelihood.py"
            if p.exists():
                found = p.resolve()
                break
        except Exception:
            continue

    if found is None:
        base = Path("/storage/emulated/0")
        if base.exists():
            try:
                for p in base.rglob("likelihood.py"):
                    found = p.resolve()
                    break
            except Exception:
                found = None

    if found is None:
        raise ModuleNotFoundError(
            "Could not find 'likelihood.py'. Looked in: " + ", ".join(str(c) for c in candidates)
        )

    spec = importlib.util.spec_from_file_location("likelihood_from_storage", str(found))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {found}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to load likelihood module from {found}: {e}")
    logger.info(f"Loaded 'likelihood.py' from: {found}")
    return module

# Load likelihood helper module and bind expected functions
try:
    _likelihood = _load_likelihood_module()
except Exception as e:
    logger.exception("Unable to load likelihood.py: %s", e)
    raise

# Required functions expected from likelihood.py
for fname in ("try_load_all", "log_likelihood_rotation", "log_likelihood_draco_dispersion", "log_likelihood_small_scale"):
    if not hasattr(_likelihood, fname):
        raise ImportError(f"likelihood.py is missing required function: {fname}")

try_load_all = getattr(_likelihood, "try_load_all")
log_likelihood_rotation = getattr(_likelihood, "log_likelihood_rotation")
log_likelihood_draco_dispersion = getattr(_likelihood, "log_likelihood_draco_dispersion")
log_likelihood_small_scale = getattr(_likelihood, "log_likelihood_small_scale")

# ---------------- Try optional emcee import ----------------
try:
    import emcee
except Exception:
    emcee = None
    logger.warning("emcee not available; install with `pip install emcee` for MCMC sampling.")

# ---------------- Priors & helpers ----------------
def log_prior(theta: np.ndarray) -> float:
    """
    Prior on parameter vector:
    theta = [log10_M200, c200, log10_Md, Rd, log10_Mb, log10_alpha0, log10_fM, jitter]
    Returns 0.0 for flat allowed region, -inf otherwise.
    """
    log10_M200, c200, log10_Md, Rd, log10_Mb, log10_alpha0, log10_fM, jitter = tuple(theta)
    if not (11.0 <= log10_M200 <= 13.5):
        return -np.inf
    if not (2.0 <= c200 <= 30.0):
        return -np.inf
    if not (8.0 <= log10_Md <= 11.0):
        return -np.inf
    if not (0.5 <= Rd <= 8.0):
        return -np.inf
    if not (7.0 <= log10_Mb <= 10.5):
        return -np.inf
    if not (-7.0 <= log10_alpha0 <= -5.0):
        return -np.inf
    if not (7.0 <= log10_fM <= 9.0):
        return -np.inf
    if not (0.0 <= jitter <= 50.0):
        return -np.inf
    return 0.0

def build_model_params_from_theta(theta: np.ndarray) -> Tuple[dict, float]:
    """
    Convert theta vector to the model_params dict expected by likelihood helpers,
    and return jitter in km/s.
    """
    log10_M200, c200, log10_Md, Rd, log10_Mb, log10_alpha0, log10_fM, jitter = tuple(theta)
    alpha0 = 10.0 ** float(log10_alpha0)
    fM = 10.0 ** float(log10_fM)
    model_params = {
        "halo": {"type": "nfw", "M200": 10.0 ** float(log10_M200), "c200": float(c200)},
        "disk": {"M": 10.0 ** float(log10_Md), "Rd": float(Rd)},
        # bulge.M is in solar masses here (likelihood helpers expect 'bulge' keys as used earlier)
        "bulge": {"M": 10.0 ** float(log10_Mb), "alpha0": alpha0, "fM": fM, "gamma": 0.48},
    }
    return model_params, float(jitter)

def log_probability(theta: np.ndarray,
                    rot_mw_df,
                    rot_ngc_df,
                    draco_sigma_df,
                    small_scale_df) -> float:
    """
    Full log-probability = log_prior + log_likelihoods.
    Exceptions during evaluation are caught and logged; return -inf for sampler stability.
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    model_params, jitter = build_model_params_from_theta(theta)
    try:
        ll = 0.0
        if rot_mw_df is not None:
            ll += log_likelihood_rotation(model_params, rot_mw_df, jitter_km_s=jitter)
        if rot_ngc_df is not None:
            ll += log_likelihood_rotation(model_params, rot_ngc_df, jitter_km_s=jitter)
        if draco_sigma_df is not None:
            ll += log_likelihood_draco_dispersion(model_params, draco_sigma_df)
        if small_scale_df is not None:
            ll += log_likelihood_small_scale(model_params, small_scale_df)
        return float(lp + ll)
    except Exception as e:
        # Log exception and return -inf so the sampler can continue
        logger.exception("Exception in log_probability evaluation: %s", e)
        return -np.inf

# ---------------- Runner ----------------
def run_mcmc(nwalkers: int = 24,
             nsteps: int = 1000,
             burn_in: int = 300,
             random_seed: int = 123) -> Tuple[Optional[object], Optional[np.ndarray]]:
    """
    Run MCMC using emcee. Returns (sampler, flat_samples) or (None, None) on failure.
    """
    logger.info("Searching for rotation / Draco / small-scale files (using likelihood helpers)...")
    rot_mw, rot_ngc, draco_sigma, small_scale = try_load_all()

    ndim = 8  # parameter vector length
    rng = np.random.default_rng(random_seed)

    # Center for initial walker positions (in same order as theta)
    p0_center = np.array([12.3, 10.0, 10.0, 3.0, 9.0, -6.0, 8.0, 5.0], dtype=float)

    # Reasonable scatter for each parameter (log-space sensible)
    scales = np.array([0.05, 0.5, 0.1, 0.2, 0.1, 0.02, 0.1, 0.5], dtype=float)
    p0 = p0_center + rng.normal(scale=scales, size=(nwalkers, ndim))

    if emcee is None:
        logger.error("emcee not installed. Install with `pip install emcee` and retry.")
        return None, None

    # Do not use HDF5 backend by default (avoid platform issues)
    backend = None

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(rot_mw, rot_ngc, draco_sigma, small_scale), backend=backend)

    logger.info("Running MCMC: %d walkers x %d steps ...", nwalkers, nsteps)
    t0 = time.time()
    try:
        # emcee >=3 supports progress=True; older versions ignore it
        sampler.run_mcmc(p0, nsteps, progress=True)
    except TypeError:
        sampler.run_mcmc(p0, nsteps)
    t1 = time.time()
    logger.info("MCMC finished in %.1f s", (t1 - t0))

    # try to fetch chain
    try:
        chain = sampler.get_chain()
    except Exception:
        chain = None
        logger.warning("Could not retrieve full chain via sampler.get_chain()")

    # flatten after burn-in
    try:
        flat = sampler.get_chain(discard=burn_in, thin=1, flat=True)
    except Exception:
        flat = None
        logger.warning("Could not flatten chain (get_chain with discard/flat failed)")

    # Save outputs robustly
    if flat is not None:
        flat_path = OUTPUT_DIR / "mcmc_flat_samples.npy"
        np.save(flat_path, flat)
        logger.info("Saved flattened samples -> %s", flat_path)
    else:
        logger.warning("Flattened samples not available; skipping save.")

    if chain is not None:
        chain_path = OUTPUT_DIR / "mcmc_chain.npy"
        np.save(chain_path, chain)
        logger.info("Saved full chain -> %s", chain_path)

    # Compute chi^2 and dof dynamically if possible
    if flat is not None:
        best_theta = np.median(flat, axis=0)
        best_params, best_jitter = build_model_params_from_theta(best_theta)

        chi2 = 0.0
        n_points = 0
        try:
            if rot_mw is not None:
                chi2 += -2.0 * log_likelihood_rotation(best_params, rot_mw, jitter_km_s=best_jitter)
                n_points += len(rot_mw)
            if rot_ngc is not None:
                chi2 += -2.0 * log_likelihood_rotation(best_params, rot_ngc, jitter_km_s=best_jitter)
                n_points += len(rot_ngc)
            if draco_sigma is not None:
                chi2 += -2.0 * log_likelihood_draco_dispersion(best_params, draco_sigma)
                n_points += len(draco_sigma)
            if small_scale is not None:
                chi2 += -2.0 * log_likelihood_small_scale(best_params, small_scale)
                n_points += len(small_scale)
        except Exception as e:
            logger.exception("Failed to compute chi2 at best-fit: %s", e)

        dof = max(1, int(n_points) - ndim)
        logger.info("Chi^2 = %.3f, n_points = %d, dof = %d, Chi^2/dof = %.3f", chi2, n_points, dof, chi2 / dof if dof > 0 else float("inf"))

    # Trace plot (one axis per parameter)
    labels = ["log10_M200", "c200", "log10_Mdisk", "Rd", "log10_Mbulge", "log10_alpha0", "log10_fM", "jitter"]
    if chain is not None and plt is not None:
        try:
            nsteps_run = chain.shape[1]
            fig, axes = plt.subplots(ndim, 1, figsize=(8, 2.0 * ndim), sharex=True)
            xaxis = np.arange(nsteps_run)
            for i in range(ndim):
                for w in range(chain.shape[0]):
                    axes[i].plot(xaxis, chain[w, :, i], color="C0", alpha=0.4)
                axes[i].set_ylabel(labels[i])
            axes[-1].set_xlabel("step")
            fig.tight_layout()
            trace_path = OUTPUT_DIR / "trace_plot.png"
            fig.savefig(trace_path, dpi=150)
            plt.close(fig)
            logger.info("Saved trace plot -> %s", trace_path)
        except Exception:
            logger.exception("Failed to create/save trace plot.")

    # Corner plot
    if flat is not None and corner is not None:
        try:
            fig = corner.corner(flat, labels=labels, show_titles=True, quantiles=[0.16, 0.5, 0.84])
            corner_path = OUTPUT_DIR / "corner_plot.png"
            fig.savefig(corner_path, dpi=150)
            plt.close(fig)
            logger.info("Saved corner plot -> %s", corner_path)
        except Exception:
            logger.exception("Failed to create/save corner plot.")

    # Print percentiles
    if flat is not None:
        try:
            pct = np.percentile(flat, [16, 50, 84], axis=0)
            logger.info("Parameter estimates (16/50/84 percentiles):")
            for i, lab in enumerate(labels):
                p16, p50, p84 = pct[0, i], pct[1, i], pct[2, i]
                logger.info("  %12s = %.5g (+%.5g / -%.5g)", lab, p50, p84 - p50, p50 - p16)
        except Exception:
            logger.exception("Failed to compute/print percentiles.")

    logger.info("Outputs in: %s", OUTPUT_DIR)
    return sampler, flat

if __name__ == "__main__":
    if emcee is None:
        logger.error("emcee is required to run this script. Install it with `pip install emcee`.")
        sys.exit(1)
    sampler_obj, flat_samples = run_mcmc(nwalkers=24, nsteps=1200, burn_in=300, random_seed=123)