import numpy as np
import matplotlib.pyplot as plt

T = 10

# Time axis
time = np.linspace(-T, T, 1000)
dt = time[1] - time[0]

# Unit step function
def unit_step(t):
    return np.where(t >= 0, 1, 0)

# Rectangular pulse function
def rectangular_pulse(t, T):
    return np.where((t >= -T) & (t <= T), 1, 0)

# Define signals
f_t = unit_step(time)
h_t = rectangular_pulse(time, T)
y_t = np.convolve(f_t, h_t, mode='full') * dt
conv_time = np.linspace(2 * time[0], 2 * time[-1], len(y_t))

# Plot all signals on the same figure
plt.figure(figsize=(10, 6))
plt.plot(time, f_t, label="f(t) = unit step", color='blue')
plt.plot(time, h_t, label="h(t) = rectangular pulse", color='orange')
plt.plot(conv_time, y_t, label="y(t) = f(t) * h(t)", color='green')

plt.title("Signals and Their Convolution")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

