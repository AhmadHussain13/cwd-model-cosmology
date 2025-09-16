# eot_wash_plot.py
import numpy as np
import matplotlib.pyplot as plt

# Use mathtext (built-in) instead of full LaTeX
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

# Parameters
Lc = 4e20       # m, characteristic length scale
alpha = 1.0     # unscreened coupling
alpha_screened = 1e-12  # effective screened coupling (thin-shell estimate)

# Distance range: 1 micron – 10 m
r = np.logspace(-6, 1, 200)  # m

# Yukawa strengths
alpha_yukawa_unscreened = alpha * np.exp(-r / Lc)  # ~flat at alpha=1
alpha_yukawa_screened   = alpha_screened * np.exp(-r / Lc)  # ~flat at 1e-12

# Plot
plt.figure(figsize=(6, 4))
plt.loglog(r * 1e3, alpha_yukawa_unscreened, 'b-', label=r'Unscreened $\alpha_{\rm Yukawa}\simeq 1$')
plt.loglog(r * 1e3, alpha_yukawa_screened, 'k-', label=r'Screened $\alpha_{\rm eff}\sim 10^{-12}$')

# Eöt-Wash bound
plt.axhline(1e-10, ls='--', color='red', label="Eöt-Wash bound")

# Labels and legend
plt.xlabel(r'Distance (mm)')
plt.ylabel(r'$\alpha_{\rm Yukawa}$')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.savefig('Fig_A2_eot_wash.png', dpi=300,bbox_inches='tight')
plt.close()