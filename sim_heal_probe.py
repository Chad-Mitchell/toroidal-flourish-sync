import numpy as np
from scipy.integrate import odeint

# Eq1 probe: Synthetic hypertension HRV (low initial xi=0.125) → irr lift
N = 50  # oscillators
t = np.linspace(0, 50, 500)
mu = 1.0  # coupling (tune <1 for milder)
v = 0.2
eta_i = 0.01 * np.random.randn(N)

golden = (1 + np.sqrt(5)) / 2
omega_i = golden * np.ones(N) + 0.1 * np.random.randn(N)

theta0 = 2 * np.pi * np.random.rand(N)

def dtheta_irr(theta, t):
    mean_exp = np.mean(np.exp(1j * theta))
    coupling = mu * np.imag(np.conj(mean_exp) * np.exp(1j * theta))
    shear = v * np.sin(np.mean(theta))
    return omega_i + coupling + shear + eta_i

theta_irr = odeint(dtheta_irr, theta0, t)
xi_irr = np.abs(np.mean(np.exp(1j * theta_irr), axis=1))

# Rat baseline
omega_rat = np.ones(N)
def dtheta_rat(theta, t):
    mean_exp = np.mean(np.exp(1j * theta))
    coupling = mu * np.imag(np.conj(mean_exp) * np.exp(1j * theta))
    shear = v * np.sin(np.mean(theta))
    return omega_rat + coupling + shear + eta_i

theta_rat = odeint(dtheta_rat, theta0, t)
xi_rat = np.abs(np.mean(np.exp(1j * theta_rat), axis=1))

print(f"Initial xi (low HRV proxy): {xi_irr[0]:.3f}")
print(f"Irr final xi: {xi_irr[-1]:.3f}")
print(f"Rational final xi: {xi_rat[-1]:.3f}")
lift = ((xi_irr[-1] - xi_rat[-1]) / xi_rat[-1]) * 100 if xi_rat[-1] > 0 else 0
print(f"Lift % (>20% conjecture): {lift:.1f}")

# Run: python sim_heal_probe.py → Paste your RR for real theta0.