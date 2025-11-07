import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt  # For viz if run local

# Extend heal_probe: Screen/WiFi eta sigmoid (anxiety creep)
N = 50
t = np.linspace(0, 50, 500)
mu = 1.0
v = 0.2

golden = (1 + np.sqrt(5)) / 2
omega_i = golden * np.ones(N) + 0.1 * np.random.randn(N)
theta0 = 2 * np.pi * np.random.rand(N)
eta_base = 0.01 * np.random.randn(N)

def eta_screen(t):
    return eta_base * (1 + 5 * (1 / (1 + np.exp(-0.5 * (t - 25)))))  # Sigmoid ramp

def dtheta_irr_screen(theta, t):
    mean_exp = np.mean(np.exp(1j * theta))
    coupling = mu * np.imag(np.conj(mean_exp) * np.exp(1j * theta))
    shear = v * np.sin(np.mean(theta))
    eta_t = eta_screen(t) * np.ones(N)
    return omega_i + coupling + shear + eta_t

theta_irr_screen = odeint(dtheta_irr_screen, theta0, t)
xi_irr_screen = np.abs(np.mean(np.exp(1j * theta_irr_screen), axis=1))

# Base (no screen)
def dtheta_irr_base(theta, t):
    mean_exp = np.mean(np.exp(1j * theta))
    coupling = mu * np.imag(np.conj(mean_exp) * np.exp(1j * theta))
    shear = v * np.sin(np.mean(theta))
    return omega_i + coupling + shear + eta_base

theta_irr_base = odeint(dtheta_irr_base, theta0, t)
xi_irr_base = np.abs(np.mean(np.exp(1j * theta_irr_base), axis=1))

print(f"Base final xi: {xi_irr_base[-1]:.3f}")
print(f"Screen noise final xi: {xi_irr_screen[-1]:.3f}")
drop = ((xi_irr_base[-1] - xi_irr_screen[-1]) / xi_irr_base[-1]) * 100
print(f"Xi volatility under screen (%): {drop:.1f}")  # Erosion metric

# Plot stub
plt.plot(t, xi_irr_base, label='Base Irr')
plt.plot(t, xi_irr_screen, label='Screen Noise')
plt.xlabel('Time'); plt.ylabel('ξ'); plt.legend(); plt.show()  # Local run