import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os

# ========================
# 1. Physical constants
# ========================
G = 6.6743e-11        # Gravitational constant (SI)
r0 = 1.616e-35        # Wormhole throat radius (m)
L_c = 4e20            # Yukawa cutoff scale (m)
rho_c = 8.69e-27      # Critical density (kg/m^3)
rho_exotic = -0.14 * rho_c

# ========================
# 2. Ensure output folder
# ========================
outdir = "figures/appendix_c"
os.makedirs(outdir, exist_ok=True)

# ========================
# 3. Exotic density profile
# ========================
r = np.logspace(-34, 20, 200)
rho = -r0**2 / (8 * np.pi * G * r**4)

plt.figure(figsize=(8, 6))
plt.loglog(r, -rho, label=r'$|\rho(r)| = r_0^2 / (8 \pi G r^4)$')
plt.axhline(-rho_exotic, color='red', linestyle='--',
            label=r'$\rho_{\rm exotic} = -1.22 \times 10^{-27}$ kg/m³')
plt.xlabel(r'Radius $r$ (m)')
plt.ylabel(r'$|\rho|$ (kg/m³)')
plt.title('Exotic Matter Density vs. Radius')
plt.legend()
plt.grid(True)
plt.savefig(f"{outdir}/rho_exotic_plot.png")

# ========================
# 4. Yukawa strength
# ========================
r_yuk = np.logspace(-6, 10, 200)
alpha_Y = 1e-40 * np.exp(-r_yuk / L_c) * (1 + r_yuk / L_c)

plt.figure(figsize=(8, 6))
plt.loglog(r_yuk, alpha_Y,
           label=r'$\alpha_Y = (M_5 / M_b)\, e^{-r/L_c}\,(1 + r/L_c)$')
plt.axhline(1e-15, color='red', linestyle='--',
            label=r'Eöt-Wash Limit ($10^{-15}$)')
plt.xlabel(r'Radius $r$ (m)')
plt.ylabel(r'Yukawa Strength $\alpha_Y$')
plt.title('Yukawa Strength vs. Eöt-Wash Constraint')
plt.legend()
plt.grid(True)
plt.savefig(f"{outdir}/yukawa_limit.png")

# ========================
# 5. Integrated exotic energy
# ========================
def integrand(r):
    return 4 * np.pi * r**2 * (-r0**2 / (8 * np.pi * G * r**4))

r_int = np.logspace(-34, 20, 100)
E_int = [quad(integrand, r0, r_i)[0] for r_i in r_int]

plt.figure(figsize=(8, 6))
plt.loglog(r_int, -np.array(E_int), label='Integrated Exotic Energy')
plt.xlabel(r'Cutoff Radius $r$ (m)')
plt.ylabel(r'$-E_{\rm exotic}$ (kg)')
plt.title('Integrated Exotic Energy vs. Cutoff Radius')
plt.legend()
plt.grid(True)
plt.savefig(f"{outdir}/energy_profile.png")

print(f"✅ Plots saved in: {outdir}")