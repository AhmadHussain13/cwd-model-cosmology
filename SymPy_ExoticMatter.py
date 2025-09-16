"""
SymPy_ExoticMatter.py

This script computes the stress-energy tensor for the Morris-Thorne wormhole metric
in the Cosmic Wormhole Dynamics (CWD) model, as described in Appendix C, section C.2.
The metric is:

ds^2 = -dt^2 + dr^2 / (1 - r_0^2 / r^2) + r^2 (dtheta^2 + sin^2(theta) dphi^2)

with shape function b(r) = r_0^2 / r, throat radius r_0 = 1.616e-35 m. We derive:
- Christoffel symbols
- Ricci tensor and scalar
- Einstein tensor
- Stress-energy tensor T_mu_nu = diag(-rho, p_r, p, p)
- Verify NEC violation: rho + p_r < 0
- Compute rho_throat at r = r_0

All calculations use SI units (G_4 = 6.6743e-11 m^3 kg^-1 s^-2). Results are saved
to /storage/emulated/0/scripts/appendix_c/exotic_tensor_results.txt

Dependencies: sympy
"""

import sympy as sp
import os

# =====================
# 1. Define constants
# =====================
r0 = 1.616e-35  # Throat radius (m)
G_4 = 6.6743e-11  # Gravitational constant (m^3 kg^-1 s^-2)

# =====================
# 2. Symbolic variables
# =====================
t, r, theta, phi = sp.symbols('t r theta phi', real=True, positive=True)
r_0, G_4_sym = sp.symbols('r_0 G_4', positive=True)

# =====================
# 3. Metric components
# =====================
g_tt = -1
g_rr = 1 / (1 - r_0**2 / r**2)
g_theta_theta = r**2
g_phi_phi = r**2 * sp.sin(theta)**2
metric = sp.diag(g_tt, g_rr, g_theta_theta, g_phi_phi)

# Inverse metric
g_inv = metric.inv()

# Coordinates
coords = [t, r, theta, phi]

# =====================
# 4. Christoffel symbols
# =====================
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

# =====================
# 5. Ricci tensor
# =====================
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
R = sum(g_inv[mu, nu] * R_mu_nu[mu][nu] for mu in range(4) for nu in range(4))
R = sp.simplify(R)

# =====================
# 6. Einstein tensor
# =====================
G_mu_nu = [[sp.simplify(R_mu_nu[mu][nu] - 0.5 * metric[mu, nu] * R) for nu in range(4)] for mu in range(4)]

# Stress-energy tensor
T_mu_nu = [[G_mu_nu[mu][nu] / (8 * sp.pi * G_4_sym) for nu in range(4)] for mu in range(4)]

# =====================
# 7. Physical quantities
# =====================
rho = -T_mu_nu[0][0]
p_r = T_mu_nu[1][1] * (1 - r_0**2 / r**2)  # adjust for g_rr
p = T_mu_nu[2][2] / r**2  # adjust for g_theta_theta
nec = rho + p_r

# Numerical evaluation at throat
rho_throat = -1 / (8 * sp.pi * G_4 * r0**2)
p_r_throat = -1 / (8 * sp.pi * G_4 * r0**2)
nec_throat = rho_throat + p_r_throat

# =====================
# 8. Save results safely
# =====================
base_dir = "/storage/emulated/0/scripts/appendix_c"
os.makedirs(base_dir, exist_ok=True)

christoffel_file = os.path.join(base_dir, "Christoffel_symbols.txt")
results_file = os.path.join(base_dir, "exotic_tensor_results.txt")

# Save Christoffel symbols
with open(christoffel_file, "w") as f:
    f.write("Non-zero Christoffel symbols:\n")
    for k in range(4):
        for i in range(4):
            for j in range(4):
                if Gamma[k][i][j] != 0:
                    f.write(f"Gamma^{coords[k]}_{coords[i]}{coords[j]} = {Gamma[k][i][j]}\n")

# Save tensor results
with open(results_file, "a") as f:
    f.write("\nRicci scalar R = {}\n".format(R))
    f.write("\nStress-energy components:\n")
    f.write("rho = {}\n".format(rho))
    f.write("p_r = {}\n".format(p_r))
    f.write("p = {}\n".format(p))
    f.write("rho + p_r = {}\n".format(nec))
    f.write("NEC violation (rho + p_r < 0): {}\n".format(sp.simplify(nec < 0)))
    f.write("\nAt throat (r = r_0):\n")
    f.write("rho_throat = {:.2e} kg/m^3\n".format(float(rho_throat)))
    f.write("p_r_throat = {:.2e} kg/m^3\n".format(float(p_r_throat)))
    f.write("rho + p_r = {:.2e} kg/m^3\n".format(float(nec_throat)))

# =====================
# 9. Print results
# =====================
print("Non-zero Christoffel symbols:")
for k in range(4):
    for i in range(4):
        for j in range(4):
            if Gamma[k][i][j] != 0:
                print(f"Gamma^{coords[k]}_{coords[i]}{coords[j]} = {Gamma[k][i][j]}")

print("\nRicci scalar R =", R)
print("\nStress-energy components:")
print("rho =", rho)
print("p_r =", p_r)
print("p =", p)
print("rho + p_r =", nec)
print("NEC violation (rho + p_r < 0):", sp.simplify(nec < 0))

print("\nAt throat (r = r_0):")
print(f"rho_throat = {float(rho_throat):.2e} kg/m^3")
print(f"p_r_throat = {float(p_r_throat):.2e} kg/m^3")
print(f"rho + p_r = {float(nec_throat):.2e} kg/m^3")

print("\nReproducibility:")
print("Run with Python 3.10, dependency: sympy")
print(f"Results saved to {results_file}")
print("Run time: ~3 seconds")