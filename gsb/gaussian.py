import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Time axis
time = np.linspace(-10, 10, 1000)
dt = time[1] - time[0]

# Gaussian input signal
def gaussian(t, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-t**2 / (2 * sigma**2))

sigma = 2.0
f_t = gaussian(time, sigma)

# Rectangular pulse kernel
def rectangular_pulse(t, T):
    return np.where((t >= -T) & (t <= T), 1, 0)

T = 0.5
h_t = rectangular_pulse(time, T)

# Perform convolution
y_t = np.convolve(f_t, h_t, mode='full') * dt
conv_time = np.linspace(2 * time[0], 2 * time[-1], len(y_t))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time, f_t, label='Input: Gaussian', color='blue')
plt.plot(time, h_t, label='Kernel: Rectangular Pulse', color='orange')
plt.plot(conv_time, y_t, label='Output: Gaussian * Rectangular', color='green')

plt.title("Convolution of Gaussian with Rectangular Pulse")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

