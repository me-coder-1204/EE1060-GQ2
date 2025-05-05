import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Sampling range
xvals = np.arange(-20, 20, 1)

# Define sinc function (func1)
def func1(x):
    return 1 if x == 0 else np.sin(x) / x

f_x = [func1(i) for i in xvals]

# Define rectangular pulse function (func2)
T = 1  # Initial T value for the rectangle function
def func2(x, T):
    return 1 if -T <= x <= T else 0

h_x = [func2(i, T) for i in xvals]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], 'r-', label='Convolution')
line_f, = ax.plot([], [], 'b-', label='func1(x) = sinc(x)')
line_h, = ax.plot([], [], 'g-', label='func2(x) = rect(x)')
ax.set_xlim(2 * xvals[0], 2 * xvals[-1])
ax.set_ylim(-5, 5)
ax.set_title("Convolution of sinc and rect (changing T)")
ax.set_xlabel("t")
ax.set_ylabel("y(t)")
ax.grid()
ax.legend()

# Update function to modify the plot
def update(T):
    # Update the rectangular kernel with changing width
    h_x = [func2(i, T) for i in xvals]
    
    # Compute the convolution
    y = np.convolve(f_x, h_x, mode='full')
    
    # Adjust the x-axis for the convolved result
    conv_xvals = np.arange(len(y)) + 2 * xvals[0]
    
    # Update the plots
    line.set_data(conv_xvals, y)
    line_f.set_data(xvals, f_x)  # func1(x) = sinc(x)
    line_h.set_data(xvals, h_x)  # func2(x) = rect(x)
    
    ax.set_title(f"Convolution of sinc and rect (T = {T})")
    return line, line_f, line_h

# Create the animation with frames for T from 1 to 10
ani = FuncAnimation(fig, update, frames=range(1, 20), interval=500, blit=True)

ani.save('conv_ani.gif', writer='imagemagick', fps=10)

# Show the animation
plt.show()

