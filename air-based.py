import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import interp1d  

# Coefficients
a = 3  # Initial guess
b = 1.4e-4  # Initial guess
c = 4  # Initial guess

# Testbed parameters
delta_h = 0.15  # Vertical height difference (m)
d = 1.0 # Transmitter-receiver distance (m)
c_h = 3.5  # Horizontal advection velocity (m/s)
c_v = 0  # Vertical advection velocity (assumed to be negligible) (m/s)

# Correction of the transmitter-receiver distance
r = lambda d, delta_h: np.sqrt(d**2 + delta_h**2)

# Time 
t_values = np.linspace(0.01, 20, 500)  

# Model definition
def M(t, a, b, c, delta_h, d):
    r_eff = r(d, delta_h)
    return (a / np.sqrt(t)) * np.exp(-b * ((r_eff - c * t)**2) / t)

# Testbed data/measurements (input already normalized)
data = pd.read_csv('air_testbed_1m.csv')
x_values = data['Time']
y_values = data['normal']

# Interpolation testbed data 
interp_func = interp1d(x_values, y_values, kind='linear', fill_value='extrapolate')
y_values_interpolated = interp_func(t_values)

# Define a residual function for least squares curve fitting
def residuals(params, t, observed_data, delta_h, d):
    a, b, c = params
    model_values = M(t, a, b, c, delta_h, d)
    return model_values / np.max(model_values) - observed_data  # Normalize model

# Initial guess for coefficients 
initial_guess = [a, b, c]

# Least squares curve fitting
result = least_squares(residuals, initial_guess, args=(t_values, y_values_interpolated, delta_h, d))

# Extract optimized parameters
a_opt, b_opt, c_opt = result.x
print(f"Coefficients: a = {a_opt}, b = {b_opt}, c = {c_opt}")

M_fitted = M(t_values, a_opt, b_opt, c_opt, delta_h, d)
M_fitted_normalized = M_fitted / np.max(M_fitted)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_values, M_fitted_normalized, label="Fitted theoretical model", color='blue')
plt.plot(t_values, y_values_interpolated, label="Interpolated testbed data", color='red', linestyle='--')
plt.xlabel("Time (t) [s]", fontsize=12)
plt.ylabel("Normalized CIR", fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()
