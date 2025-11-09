import numpy as np
from scipy.integrate import odeint

# Eq1 fractal probe: Weak individual (irr mixing) → hybrid dyad → strong community (rat variance)
N_ind = 10; N_dyad = 20; N_comm = 50  # Scale nodes
t = np.linspace(0, 50, 500)
mu_weak = 0.5; mu_hybrid = 1.3; mu_strong = 2.0
v = 0.2; eta_i = 0.01 * np.random.randn(max(N_ind, N_dyad, N_comm))

golden = (1 + np.sqrt(5)) / 2
omega_irr = golden * np.ones(max(N_ind, N_dyad, N_comm)) + 0.1 * np.random.randn(max(N_ind, N_dyad, N_comm))
omega_rat = np.ones(max(N_ind, N_dyad, N_comm))

def dtheta_regime(theta, t, mu, omega):
    mean_exp = np.mean(np.exp(1j * theta))
    coupling = mu * np.imag(np.conj(mean_exp) * np.exp(1j * theta))
    shear = v * np.sin(np.mean(theta))
    return omega + coupling + shear + eta_i[:len(theta)]

# Weak irr individual
theta_weak = odeint(lambda theta, t: dtheta_regime(theta, t, mu_weak, omega_irr), 2 * np.pi * np.random.rand(N_ind), t)
xi_weak = np.abs(np.mean(np.exp(1j * theta_weak), axis=1))[-1]

# Hybrid dyad
theta_hybrid = odeint(lambda theta, t: dtheta_regime(theta, t, mu_hybrid, omega_irr), 2 * np.pi * np.random.rand(N_dyad), t)
xi_hybrid = np.abs(np.mean(np.exp(1j * theta_hybrid), axis=1))[-1]

# Strong rat community
theta_strong = odeint(lambda theta, t: dtheta_regime(theta, t, mu_strong, omega_rat), 2 * np.pi * np.random.rand(N_comm), t)
xi_strong = np.abs(np.mean(np.exp(1j * theta_strong), axis=1))[-1]

print(f"Weak irr xi (individual): {xi_weak:.3f}")
print(f"Hybrid xi (dyad/family): {xi_hybrid:.3f}")
print(f"Strong rat xi (community): {xi_strong:.3f}")
mi_bonus = ((xi_hybrid - xi_weak) / xi_weak) * 100 if xi_weak > 0 else 0
print(f"Hybrid MI bonus % (+15-25% conjecture): {mi_bonus:.1f}")
variance_lift = ((xi_strong - xi_hybrid) / xi_hybrid) * 100 if xi_hybrid > 0 else 0
print(f"Strong variance lift % (+1700% max): {variance_lift:.1f}")

# Run: python sim_fractal.py → Scale your HRV data for fractal ξ.