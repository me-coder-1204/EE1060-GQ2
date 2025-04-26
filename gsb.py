import numpy as np
import matplotlib.pyplot as plt

T = 25
t0 = -10

# Function to find sinc(x)
def func1(x):
    return 1 if x == 0 else np.sin(x) / x

# Rectangular kernel
def func2(x):
    return 1 if -T <= x <= T else 0

xvals = [i for i in range(-3*T, 3*T)]
f_x = [func1(i) for i in xvals]
h_x = [func2(i) for i in xvals]

# Convolution
x = np.convolve(f_x, h_x)
conv_xvals = np.arange(len(x)) + 2 * xvals[0]
mini = min(x)

# Define the given y-value
y_value = min(x)  # example y-value (you can change this)

# Find corresponding x-value in convolution result
# Searching for the closest value to the given y-value in the convolution result
closest_index = np.argmin(np.abs(np.array(x) - y_value))
closest_x = conv_xvals[closest_index]
closest_y = x[closest_index]

# Plot
plt.scatter(closest_x, closest_y, color='blue', label=f'Point ({closest_x}, {closest_y})')
plt.plot(xvals, f_x, label='Function f(t)')
plt.plot(xvals, h_x, label='Kernel h(t)')
plt.plot(conv_xvals, x, label='Convoluted function y(t)', color='red')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title(f'Convolution of f(t) and h(t) for T = {T}')
plt.legend()
plt.grid(True)
plt.savefig('Conv_sinc.png')
plt.show()

