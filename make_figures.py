import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
G = 6.6743e-11
r_s = 1.54e+22
rho_s = 3.7e-23
M_5 = 3e+41
L = 4.63e+20
R_vir = 4.63e+22
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
