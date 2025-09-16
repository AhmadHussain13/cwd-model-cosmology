# rho_eff_plot.py
import numpy as np
import matplotlib.pyplot as plt

# Force matplotlib to use mathtext only (no LaTeX dependency)
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

# Parameters
M5 = 3e41  # kg
L = 15  # kpc
kpc_to_m = 3.086e19  # m/kpc
r = np.linspace(0.1, 50, 100)  # kpc
rho_eff = (M5 / (4 * np.pi * (L * kpc_to_m)**3 * (r * kpc_to_m))) \
          * np.exp(-r/L) * (1 - r/L)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(r, rho_eff, label=r'$\rho_{\mathrm{eff}}(r)$')
plt.axvline(L, ls='--', color='red', label=r'$r=L$')
plt.xlabel(r'$r \, (\mathrm{kpc})$')
plt.ylabel(r'$\rho_{\mathrm{eff}} \, (\mathrm{kg/m^3})$')
plt.legend()
plt.grid(True)
plt.savefig('Fig_A1_rho_eff.png', dpi=300, bbox_inches='tight')
plt.close()