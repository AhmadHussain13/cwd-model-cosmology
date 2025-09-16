# cwd-model-cosmology
# cwd-model/cosmology

## Overview

This repository contains the code, data, and documentation for reproducing the results in the preprint "Cosmic Wormhole Dynamics: A Geometric Model for Cosmic Expansion" by Ahmad Hussain (MNRAS preprint, September 16, 2025). The Cosmic Wormhole Dynamics (CWD) model describes the universe as a 4D hypersurface embedded in a 5D wormhole spacetime, where late-time acceleration is driven by the dynamical evolution of the wormhole throat radius. It eliminates the need for a cosmological constant and provides a Yukawa-like modification to gravity that explains galactic rotation curve anomalies without dark matter halos.

The model is validated against astrophysical and cosmological data, including Milky Way and NGC 3198 rotation curves, Draco dwarf kinematics, Coma and Abell cluster velocity dispersions, Planck 2018 CMB spectra, Pantheon+ supernovae, and DESI BAO/quasar clustering. An emcee MCMC analysis with parameters like warp factor \(k\), 5D mass scale \(M_{5,\text{global}}\), scalar exponent \(\kappa\), coupling \(\alpha_0\), and scaling \(\gamma\) yields robust posteriors (e.g., \(k = (2.16 \pm 0.5) \times 10^{-21}\) m\(^{-1}\)).

This repo supports theoretical derivations (e.g., Einstein tensor, Klein-Gordon equation, exotic matter), numerical examples, and predictions (e.g., CMB residuals ) 

## Features

- **Theoretical Derivations**: 5D Einstein equations, Ricci tensor, Christoffel symbols, Klein-Gordon equation in curved spacetime, Weyl projection to 4D effective potential, exotic matter Casimir estimate, wormhole stability analysis.
- **Numerical Computations**: Slow-roll scalar field evolution, rotation curve fits, velocity dispersions, lensing convergence maps, integrated exotic energy budget.
- **Observational Validation**: Chi-squared tests, MCMC with emcee (100 walkers, 5000 steps, 1000 burn-in), AIC/BIC comparisons to ΛCDM.
- **Predictions**: Modified CMB power spectra, high-z quasar clustering, substructure counts, void growth rates.
- **Appendices Support**: Code for Appendices A (Einstein tensor), B (scalar equation), C (Casimir), D (Friedmann projection), E (5D gravitational effects, geodesics, Weyl potential), F (scalar dynamics, Klein-Gordon, stability), G (exotic matter, stability, lensing, constraints), H (likelihood, MCMC, constraints), I (numerical examples), J (quasar constraints), K (cluster dispersions).
- **Tools**: Python 3.8, SymPy for symbolic math, NumPy/SciPy for numerics, modified CLASS v2.9 for cosmology, emcee v3.1 for MCMC, Matplotlib for plots.

## Installation

1. **Prerequisites**: Python 3.8+, C compiler (for CLASS).
2. **Dependencies**: 
   ```
   pip install numpy scipy sympy emcee matplotlib
   ```
3. **CLASS v2.9 Setup**:
   - Download from http://class-code.net.
   - Apply patch from `class_patch/cwd_patch.diff`:
     ```
     cd class
     patch -p1 < ../class_patch/cwd_patch.diff
     make
     ```
   - Add CLASS bin to PATH.
4. **Clone Repo**:
   ```
   git clone https://github.com/cwd-model/cosmology.git
   cd cosmology
   ```
5. **Verify**:
   ```
   python test_installation.py
   ```

## Usage

### Directory Structure
- `scripts/`: Main Python scripts.
  - `sympy_christoffel.py`: Computes Christoffel symbols (Appendix E).
  - `sympy_kleingordon.py`: Derives/solves Klein-Gordon equation (Appendix F).
  - `sympy_exoticmatter.py`: Exotic matter derivations (Appendix G).
  - `rotation_curves.py`: Fits curves for Milky Way, NGC 3198, Draco (Appendix I).
  - `cluster_dispersions.py`: Computes for Coma/Abell (Appendix K).
  - `cmb_analysis.py`: CMB spectra vs. Planck (Appendix F/H).
  - `mcmc_fitting.py`: emcee MCMC (Appendix H).
  - `lensing_maps.py`: Convergence κ(θ) (Appendix G).
  - `stability_modes.py`: Perturbation analysis (Appendix G).
  - `rho_eff_plot.py`: Effective density plots (Appendix E/G).
  - `eot_wash_plot.py`: Yukawa constraints (Appendix G).
- `data/`: Datasets (THINGS, Pantheon+, Planck 2018, DESI BAO/quasar, field_evolution.txt, energy_density.txt).
- `class_patch/`: CLASS modifications for scalar field.
- `models/`: Baryonic profiles (exponential disks, Plummer).
- `results/`: Generated outputs (plots, chains, e.g., mcmc_chain.h5).
- `notebooks/`: Jupyter notebooks for appendices (e.g., SymPy_Christoffel.ipynb, SymPy_Weyl.ipynb, SymPy_KleinGordon_CWD_v1.ipynb, SymPy_ExoticMatter.ipynb).

### Examples
- Rotation Curves:
  ```
  python scripts/rotation_curves.py --galaxy ngc3198 --r_max 15e3 --output results/ngc3198_fit.png
  ```
- MCMC:
  ```
  python scripts/mcmc_fitting.py --nwalkers 100 --nsteps 5000 --burnin 1000 --datasets all --output results/posteriors.h5
  ```
- Klein-Gordon Solution (Appendix F):
  ```
  python scripts/sympy_kleingordon.py --r_throat 1.616e-35 --omega 1e-20 --m 1e-35 --output results/field_evolution.txt
  ```
- Exotic Density Plot (Appendix G):
  ```
  python scripts/exotic_density_plot.py --r0 1.616e-35 --plot_type rho_eff --output results/figure_g1.png
  ```
- Stability Modes (Appendix G):
  ```
  python scripts/stability_modes.py --perturb_mode radial --output results/figure_g2.png
  ```
- Lensing Convergence (Appendix G):
  ```
  python scripts/lensing_maps.py --cluster bullet --theta_max 1e-4 --output results/figure_g1_kappa.png
  ```

Refer to script docstrings or `--help` for options. Numerical examples from Appendix I (e.g., Draco σ_v at r=0.5 kpc) are in `scripts/numerical_examples.py`.

## Citation

Cite the preprint:
- Hussain, A. (2025). "Cosmic Wormhole Dynamics: A Geometric Model for Cosmic Expansion." MNRAS, preprint. E-mail: ah0795223@gmail.com.

## License

MIT License (see LICENSE).

## Contributing

Fork, branch, pull request. Open issues for discussions.

## Contact

ah0795223@gmail.com

## Acknowledgments

Thanks to open-source tools (SymPy, emcee, CLASS) and data providers (Planck, Pantheon+, DESI, THINGS). Commit hash: [insert commit hash]. 
