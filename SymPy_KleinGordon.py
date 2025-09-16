import sympy as sp
from scipy.integrate import solve_ivp
import numpy as np

# -----------------------------
# Symbolic Klein-Gordon Setup
# -----------------------------

# Symbols
r, t = sp.symbols('r t', real=True)
r0, omega, m, phi_0 = sp.symbols('r0 omega m phi_0', real=True, positive=True)

# Unknown function and ansatz
u = sp.Function('u')(r)
phi = phi_0 * sp.exp(-sp.I * omega * t) * u

# Metric factor (spherical, symbolic)
grr = 1 / (1 - r0**2 / r)
sqrt_g = r**2

# Radial KG term
kg_term_r = (1 / sqrt_g) * sp.diff(sqrt_g * grr * sp.diff(phi, r), r)
kg_radial = kg_term_r + omega**2 * phi - m**2 * phi
kg_radial = sp.expand(kg_radial)

# Extract coefficients
u_2nd = sp.diff(u, r, 2)
u_1st = sp.diff(u, r)
coeff_2nd = kg_radial.coeff(u_2nd)
coeff_1st = kg_radial.coeff(u_1st)
coeff_0 = kg_radial.coeff(u)

# Print symbolic radial equation
print("Symbolic Radial Klein-Gordon Equation:")
eq = f"{u_2nd} + ({sp.simplify(coeff_1st)}) {u_1st} + ({sp.simplify(coeff_0)}) {u} = 0"
print(eq)

# -----------------------------
# Numerical ODE Setup
# -----------------------------

# Function P(r) from symbolic coefficient of u'(r)
def P(r_val, r0_val):
    return (2*r_val - 3*r0_val**2) / (r_val**2 - 2*r_val*r0_val**2 + r0_val**4)

# Function Q(r) from coefficient of u(r)
def Q(omega_val, m_val):
    return omega_val**2 - m_val**2

# ODE system
def kg_ode(r_val, u_vec, params):
    r0_val, omega_val, m_val = params
    u0, u1 = u_vec
    du0_dr = u1
    du1_dr = -P(r_val, r0_val)*u1 - Q(omega_val, m_val)*u0
    return [du0_dr, du1_dr]

# Numerical parameters
r0_val = 1.616e-35
omega_val = 1e-20
m_val = 1e-35
phi0_val = 1e-10

params = [r0_val, omega_val, m_val]
r_span = [r0_val * 1.1, 1e20]
u0_vec = [1.0, 0.0]

# Solve ODE
sol = solve_ivp(kg_ode, r_span, u0_vec, args=(params,), dense_output=True, method='RK45')
r_vals = np.logspace(np.log10(r_span[0]*1.1), np.log10(r_span[1]), 1000)
u_vals = sol.sol(r_vals)[0]

# Compute energy density
rho_phi_vals = 0.5 * (omega_val**2 * phi0_val**2 * u_vals**2 + m_val**2 * phi0_val**2 * u_vals**2)

# Save for plotting
np.savetxt('field_evolution.txt', np.column_stack((r_vals, u_vals)))
np.savetxt('energy_density.txt', np.column_stack((r_vals, rho_phi_vals)))

# Print sample output
idx_1e19 = np.argmin(np.abs(r_vals - 1e19))
print(f"Field amplitude at r = 1e19 m: {u_vals[idx_1e19]:.4e}")
print(f"Energy density at r = 1e19 m: {rho_phi_vals[idx_1e19]:.4e} kg/m^3")