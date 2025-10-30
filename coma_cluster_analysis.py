# Third attempt: avoid f-strings when creating the script files to prevent brace conflicts.
from pathlib import Path
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import textwrap
import os

# Save to external storage so it's visible in File Manager
outdir = Path(os.environ.get("EXTERNAL_STORAGE", "/sdcard")) / "coma_appendix_outputs"
outdir.mkdir(parents=True, exist_ok=True)


# Parameters
G = 6.6743e-11
M = 2.4e45
R_vir = 4.63e22
r_s = 1.54e22
rho_s = 3.7e-23
M_5 = 3e41
L = 4.63e20

# Templates with placeholders
coma_virial_template = textwrap.dedent("""\
    import numpy as np
    G = __G__
    def sigma_virial(M, R, alpha=3.0):
        \"\"\"General virial-like relation:
        sigma^2 = G M / (alpha R)
        Common choices: alpha=3 (spherical isotropic), alpha=5 (uniform sphere)
        \"\"\"
        return np.sqrt(G * M / (alpha * R))
    if __name__ == "__main__":
        M = __M__
        R = __R__
        print("Virial sigma (alpha=3.0): {:.1f} km/s".format(sigma_virial(M,R,alpha=3.0)/1e3))
        print("Virial sigma (alpha=5.0): {:.1f} km/s".format(sigma_virial(M,R,alpha=5.0)/1e3))
""")

coma_jeans_template = textwrap.dedent("""\
    import numpy as np
    from scipy.integrate import quad
    G = __G__
    r_s = __r_s__
    rho_s = __rho_s__
    R_vir = __R__
    def rho_nfw(r):
        x = r / r_s
        return rho_s / (x * (1.0 + x)**2)
    def M_nfw_enclosed(r):
        x = r / r_s
        return 4.0 * np.pi * rho_s * r_s**3 * (np.log(1.0 + x) - x / (1.0 + x))
    def dPhi_dr(r):
        r = max(r, 1e-300)
        return G * M_nfw_enclosed(r) / (r**2)
    def sigma_r2(r):
        nu = rho_nfw(r)
        if nu <= 0:
            return 0.0
        integrand = lambda s: rho_nfw(s) * dPhi_dr(s)
        integral = quad(integrand, r, R_vir, limit=200)[0]
        integral = max(integral, 0.0)
        return integral / nu
    if __name__ == "__main__":
        r_test = r_s
        print("sigma_r(r_s) [km/s] = {:.1f}".format(np.sqrt(sigma_r2(r_test))/1e3))
""")

make_figures_template = textwrap.dedent("""\
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import quad
    G = __G__
    r_s = __r_s__
    rho_s = __rho_s__
    M_5 = __M_5__
    L = __L__
    R_vir = __R__
    def rho_nfw(r):
        x = r / r_s
        return rho_s / (x * (1.0 + x)**2)
    def M_nfw_enclosed(r):
        x = r / r_s
        return 4.0 * np.pi * rho_s * r_s**3 * (np.log(1.0 + x) - x / (1.0 + x))
    def rho_5d(r):
        return (M_5/(4.0*np.pi*L**3)) * np.exp(-r/L) / (r + 1e-300) * (1.0 - r/L)
    r = np.logspace(np.log10(r_s/100.0), np.log10(R_vir), 500)
    rho_b = rho_nfw(r)
    rho5 = rho_5d(r)
    plt.figure()
    plt.loglog(r, rho_b)
    plt.loglog(r, np.abs(rho5))
    plt.axvline(L, ls='--')
    plt.xlabel('r [m]')
    plt.ylabel('rho [kg m^-3]')
    plt.title('Density Profiles')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.savefig('/mnt/data/coma_appendix_outputs/density_profiles.png', dpi=300)
    plt.close()
    M_enc = M_nfw_enclosed(r)
    M5_enc = M_5 * (r / L**2) * np.exp(-r / L)
    plt.figure()
    plt.loglog(r, M_enc)
    plt.loglog(r, np.abs(M5_enc))
    plt.axvline(R_vir, ls='--')
    plt.xlabel('r [m]')
    plt.ylabel('M_enc [kg]')
    plt.title('Enclosed Mass')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.savefig('/mnt/data/coma_appendix_outputs/enclosed_mass.png', dpi=300)
    plt.close()
""")

# Replace placeholders
coma_virial_code = coma_virial_template.replace("__G__", repr(G)).replace("__M__", repr(M)).replace("__R__", repr(R_vir))
coma_jeans_code = (coma_jeans_template.replace("__G__", repr(G))
                                   .replace("__r_s__", repr(r_s))
                                   .replace("__rho_s__", repr(rho_s))
                                   .replace("__R__", repr(R_vir)))
make_figures_code = (make_figures_template.replace("__G__", repr(G))
                                     .replace("__r_s__", repr(r_s))
                                     .replace("__rho_s__", repr(rho_s))
                                     .replace("__M_5__", repr(M_5))
                                     .replace("__L__", repr(L))
                                     .replace("__R__", repr(R_vir)))

# Write scripts
script_dir = outdir / "scripts"
script_dir.mkdir(exist_ok=True)
(coma_virial_path := script_dir / "coma_virial.py").write_text(coma_virial_code)
(coma_jeans_path := script_dir / "coma_jeans.py").write_text(coma_jeans_code)
(make_figures_path := script_dir / "make_figures.py").write_text(make_figures_code)

# Numerical computations and figures (same as before but with guards)
r_min = r_s / 100.0
r_vals = np.logspace(np.log10(r_min), np.log10(R_vir), 300)

def rho_nfw_local(r):
    x = r / r_s
    return rho_s / (x * (1.0 + x)**2)

rho_b_vals = rho_nfw_local(r_vals)
rho5_vals = (M_5/(4.0*np.pi*L**3)) * np.exp(-r_vals / L) / (r_vals + 1e-300) * (1.0 - r_vals / L)

plt.figure()
plt.loglog(r_vals, rho_b_vals)
plt.loglog(r_vals, np.abs(rho5_vals))
plt.axvline(L, ls='--')
plt.xlabel('r [m]')
plt.ylabel('rho [kg m^-3]')
plt.title('Density Profiles — NFW (baryonic) and 5D (abs value shown)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.savefig(outdir / "density_profiles.png", dpi=300)
plt.close()

M_enc_vals = 4.0 * np.pi * rho_s * r_s**3 * (np.log(1.0 + r_vals/r_s) - (r_vals/r_s) / (1.0 + r_vals/r_s))
M5_enc_vals = M_5 * (r_vals / L**2) * np.exp(-r_vals / L)
plt.figure()
plt.loglog(r_vals, M_enc_vals)
plt.loglog(r_vals, np.abs(M5_enc_vals))
plt.axvline(R_vir, ls='--')
plt.xlabel('r [m]')
plt.ylabel('M_enc [kg]')
plt.title('Enclosed Mass — Baryonic and 5D')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.savefig(outdir / "enclosed_mass.png", dpi=300)
plt.close()

def dPhi_dr_local(r):
    r = max(r, 1e-300)
    return G * (4.0 * np.pi * rho_s * r_s**3 * (np.log(1.0 + r/r_s) - (r/r_s)/(1.0 + r/r_s))) / (r**2)

def sigma_r2_numeric_guarded(r):
    nu_r = rho_nfw_local(r)
    if nu_r <= 0:
        return 0.0
    integrand = lambda s: rho_nfw_local(s) * dPhi_dr_local(s)
    integral = quad(integrand, r, R_vir, limit=200, epsabs=0, epsrel=1e-6)[0]
    integral = max(integral, 0.0)
    return integral / nu_r

sigma_r_vals = np.sqrt([max(0.0, sigma_r2_numeric_guarded(rr)) for rr in r_vals])
plt.figure()
plt.loglog(r_vals, sigma_r_vals/1e3)
plt.xlabel('r [m]')
plt.ylabel('sigma_r(r) [km/s]')
plt.title('Radial Velocity Dispersion sigma_r(r)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.savefig(outdir / "sigma_r_profile.png", dpi=300)
plt.close()

# Projected LOS dispersion
R_proj = np.logspace(np.log10(r_min/5.0), np.log10(R_vir*0.98), 80)
Sigma_vals = np.zeros_like(R_proj)
sigma_los2_vals = np.zeros_like(R_proj)

def Sigma_of_R(Rp):
    integrand = lambda r: rho_nfw_local(r) * r / np.sqrt(r**2 - Rp**2)
    val = 2.0 * quad(integrand, Rp, R_vir, limit=200, epsabs=0, epsrel=1e-6)[0]
    return val

def sigma_los2_of_R(Rp):
    SigmaR = Sigma_of_R(Rp)
    if SigmaR <= 0:
        return 0.0
    integrand = lambda r: rho_nfw_local(r) * sigma_r2_numeric_guarded(r) * r / np.sqrt(r**2 - Rp**2)
    val = 2.0 * quad(integrand, Rp, R_vir, limit=200, epsabs=0, epsrel=1e-6)[0]
    return val / SigmaR

for i, Rp in enumerate(R_proj):
    Sigma_vals[i] = Sigma_of_R(Rp)
    sigma_los2_vals[i] = sigma_los2_of_R(Rp)

sigma_los_vals = np.sqrt(np.maximum(0.0, sigma_los2_vals))
plt.figure()
plt.loglog(R_proj, sigma_los_vals/1e3)
plt.xlabel('Projected radius R [m]')
plt.ylabel('sigma_los(R) [km/s]')
plt.title('Projected Line-of-Sight Velocity Dispersion sigma_los(R)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.savefig(outdir / "sigma_los_profile.png", dpi=300)
plt.close()

# Global sigma_v
weights = 2.0 * np.pi * R_proj
num = np.trapz(weights * Sigma_vals * sigma_los2_vals, R_proj)
den = np.trapz(weights * Sigma_vals, R_proj)
sigma_v_global = np.sqrt(num / den)

# kappa
Sigma_crit = 1e9
kappa_vals = Sigma_vals / Sigma_crit
plt.figure()
plt.loglog(R_proj, kappa_vals)
plt.xlabel('Projected radius R [m]')
plt.ylabel('kappa (Sigma/Sigma_crit)')
plt.title('Lensing Convergence kappa(R) — fiducial Sigma_crit')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.savefig(outdir / "kappa_profile.png", dpi=300)
plt.close()

# Save summary
summary_text = ("Coma appendix computed outputs\\n"
                f"Output folder: {outdir}\\n"
                "Parameters:\\n"
                f"  M = {M:.3e} kg\\n"
                f"  R_vir = {R_vir:.3e} m\\n"
                f"  r_s = {r_s:.3e} m\\n"
                f"  rho_s = {rho_s:.3e} kg/m^3\\n"
                f"  M_5 = {M_5:.3e} kg\\n"
                f"  L = {L:.3e} m\\n\\n"
                "Key results:\\n"
                f"  Virial estimate (alpha=3.0): sigma_v = {np.sqrt(G*M/(3.0*R_vir))/1e3:.1f} km/s\\n"
                f"  Virial estimate (alpha=5.0): sigma_v = {np.sqrt(G*M/(5.0*R_vir))/1e3:.1f} km/s\\n"
                f"  Jeans projected global sigma_v (surface-weighted) = {sigma_v_global/1e3:.1f} km/s\\n\\n"
                "Notes:\\n"
                "  - Sigma_crit used for lensing kappa(R) is a fiducial value = 1e9 kg/m^2. Replace with exact value computed from lens/source distances for precise lensing predictions.\\n"
                "  - Integrals use finite upper limit R_vir (virial radius).\\n"
                "  - rho_s value was corrected and used consistently.\\n"
               )
(outdir / "summary.txt").write_text(summary_text)

# copy scripts to outdir root
for p in [coma_virial_path, coma_jeans_path, make_figures_path]:
    dest = outdir / p.name
    dest.write_text(p.read_text())

print("=== Computation complete (pass 3) ===")
print(f"Saved figures to: {outdir}")
print(f"Virial sigma (alpha=3.0) = {np.sqrt(G*M/(3.0*R_vir))/1e3:.1f} km/s")
print(f"Virial sigma (alpha=5.0) = {np.sqrt(G*M/(5.0*R_vir))/1e3:.1f} km/s")
print(f"Jeans projected global sigma_v (surface-weighted) = {sigma_v_global/1e3:.1f} km/s")

print()
print("Files you can download:")
for fn in ["density_profiles.png","enclosed_mass.png","sigma_r_profile.png","sigma_los_profile.png","kappa_profile.png","coma_virial.py","coma_jeans.py","make_figures.py","summary.txt"]:
    print(f"- /mnt/data/coma_appendix_outputs/{fn}")

# preview
plt.figure()
plt.loglog(R_proj, sigma_los_vals/1e3)
plt.xlabel('Projected radius R [m]')
plt.ylabel('sigma_los(R) [km/s]')
plt.title('Projected Line-of-Sight Velocity Dispersion sigma_los(R) — preview')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.show()
