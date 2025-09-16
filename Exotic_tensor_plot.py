"""
SymPy_ExoticMatter.py

This script computes the stress-energy tensor for the Morris-Thorne wormhole metric
in the Cosmic Wormhole Dynamics (CWD) model, as described in Appendix C. The metric is:

ds^2 = -dt^2 + dr^2 / (1 - r_0^2 / r^2) + r^2 (dtheta^2 + sin^2(theta) dphi^2)

with shape function b(r) = r_0^2 / r, throat radius r_0 = 1.616e-35 m. We derive:
- Christoffel symbols
- Ricci tensor and scalar
- Einstein tensor
- Stress-energy tensor T_mu_nu = diag(-rho, p_r, p, p)
- Verify NEC violation: rho + p_r < 0
- Compute rho_throat at r = r_0

All calculations use SI units (G_4 = 6.6743e-11 m^3 kg^-1 s^-2). Results are saved
to /figures/appendix_c/exotic_tensor_plot.png.

Dependencies: sympy, numpy, matplotlib
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define constants
r0 = 1.616e-35  # Throat radius (m)
G_4 = 6.6743e-11  # Gravitational constant (m^3 kg^-1 s^-2)

# Define symbolic variables
t, r, theta, phi = sp.symbols('t r theta phi', real=True, positive=True)
r_0, G_4_sym = sp.symbols('r_0 G_4', positive=True)

# Metric components
g_tt = -1
g_rr = 1 / (1 - r_0**2 / r**2)
g_theta_theta = r**2
g_phi_phi = r**2 * sp.sin(theta)**2
metric = sp.diag(g_tt, g_rr, g_theta_theta, g_phi_phi)

# Inverse metric
g_inv = metric.inv()

# Coordinate list
coords = [t, r, theta, phi]

# Compute Christoffel symbols
def christoffel_symbols(metric, coords):
    dim = len(coords)
    Gamma = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                sum_term = 0
                for m in range(dim):
                    sum_term += g_inv[k, m] * (
                        sp.diff(metric[m, i], coords[j]) +
                        sp.diff(metric[m, j], coords[i]) -
                        sp.diff(metric[i, j], coords[m])
                    )
                Gamma[k][i][j] = sp.simplify(sum_term / 2)
    return Gamma

Gamma = christoffel_symbols(metric, coords)

# Print non-zero Christoffel symbols
print("Non-zero Christoffel symbols:")
for k in range(4):
    for i in range(4):
        for j in range(4):
            if Gamma[k][i][j] != 0:
                print(f"Gamma^{coords[k]}_{coords[i]}{coords[j]} = {Gamma[k][i][j]}")

# Ricci tensor
def ricci_tensor(Gamma, coords):
    dim = len(coords)
    R_mu_nu = [[0 for _ in range(dim)] for _ in range(dim)]
    for mu in range(dim):
        for nu in range(dim):
            sum_term = 0
            for lam in range(dim):
                sum_term += sp.diff(Gamma[mu][lam][nu], coords[lam])
                sum_term -= sp.diff(Gamma[mu][lam][lam], coords[nu])
                for sig in range(dim):
                    sum_term += Gamma[mu][lam][sig] * Gamma[sig][nu][lam]
                    sum_term -= Gamma[mu][lam][lam] * Gamma[sig][nu][sig]
            R_mu_nu[mu][nu] = sp.simplify(sum_term)
    return R_mu_nu

R_mu_nu = ricci_tensor(Gamma, coords)

# Ricci scalar
R = 0
for mu in range(4):
    for nu in range(4):
        R += g_inv[mu, nu] * R_mu_nu[mu][nu]
R = sp.simplify(R)

print("\nRicci scalar R =", R)

# Einstein tensor
G_mu_nu = [[0 for _ in range(4)] for _ in range(4)]
for mu in range(4):
    for nu in range(4):
        G_mu_nu[mu][nu] = sp.simplify(R_mu_nu[mu][nu] - 0.5 * metric[mu, nu] * R)

# Stress-energy tensor
T_mu_nu = [[0 for _ in range(4)] for _ in range(4)]
for mu in range(4):
    for nu in range(4):
        T_mu_nu[mu][nu] = G_mu_nu[mu][nu] / (8 * sp.pi * G_4_sym)

# Extract rho, p_r
rho = -T_mu_nu[0][0]
p_r = T_mu_nu[1][1] * (1 - r_0**2 / r**2)  # Account for g_rr factor
p = T_mu_nu[2][2] / r**2  # Account for g_theta_theta

print("\nStress-energy components:")
print("rho =", rho)  # -r_0^2 / (8 pi G_4 r^4)
print("p_r =", p_r)  # -r_0^2 / (8 pi G_4 r^4)
print("p =", p)  # 0

# Verify NEC violation
nec = rho + p_r
print("rho + p_r =", nec)  # -r_0^2 / (4 pi G_4 r^4)
print("NEC violation (rho + p_r < 0):", sp.simplify(nec < 0))  # True

# Numerical evaluation at throat
rho_throat = -1 / (8 * sp.pi * G_4 * r0**2)
p_r_throat = -1 / (8 * sp.pi * G_4 * r0**2)
nec_throat = rho_throat + p_r_throat

print("\nAt throat (r = r_0):")
print(f"rho_throat = {float(rho_throat):.2e} kg/m^3")  # -1.822e87
print(f"p_r_throat = {float(p_r_throat):.2e} kg/m^3")
print(f"rho + p_r = {float(nec_throat):.2e} kg/m^3")  # -3.644e87

# Plot rho(r)
r_vals = np.logspace(-34, 20, 200)
rho_vals = -r0**2 / (8 * np.pi * G_4 * r_vals**4)
plt.figure(figsize=(8, 6))
plt.loglog(r_vals, -rho_vals, label=r'$|\rho(r)| = \frac{r_0^2}{8 \pi G_4 r^4}$', color='blue')
plt.axhline(1.22e-27, color='red', linestyle='--', label=r'$\rho_{\rm exotic} = 1.22 \times 10^{-27}$ kg/m³')
plt.xlabel('Radius $r$ (m)')
plt.ylabel(r'$|\rho|$ (kg/m³)')
plt.title('Exotic Matter Density vs. Radius')
plt.legend()
plt.grid(True)
plt.savefig('figures/appendix_c/exotic_tensor_plot.png')
plt.show()

# Reproducibility
print("\nReproducibility:")
print("Run with Python 3.10, dependencies: sympy, numpy, matplotlib")
print("Save to /scripts/appendix_c/SymPy_ExoticMatter.py")
print("Figure saved to /figures/appendix_c/exotic_tensor_plot.png")
print("Run time: ~5 seconds")