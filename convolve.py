import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cmath

def convolve(f, h):
    X = np.zeros(len(f)+len(h)-1)
    for i in range(len(f)+len(h)-1):
        s = 0
        for j in range(len(f)):
            if i>=j and i < j + len(h):
                s += x[j] * h[i-j]
        X[i] = s

    return X

def fft(x):
    n = len(x)
    if n==1:
        return x
    om = cmath.exp(-2*np.pi*complex(0, 1)/n)
    Pe = x[0::2]
    Po = x[1::2]
    ye, yo = fft(Pe), fft(Po)

    y = np.zeros(n).astype(complex)
    for i in range(int(n/2)):
        y[i] = ye[i] + om**i * yo[i]
        y[i+int(n/2)] = ye[i] - om**i * yo[i]
    return y

def ifft(x):
    n = len(x)
    if n==1:
        return x
    om = cmath.exp(2*np.pi*complex(0, 1)/n)/n
    Pe = x[::2]
    Po = x[1::2]
    ye, yo = ifft(Pe), ifft(Po)

    y = np.zeros(n).astype(complex)
    for i in range(int(n/2)):
        y[i] = ye[i] + om**i * yo[i]
        y[i+int(n/2)] = ye[i] - om**i * yo[i]
    return y

n = 10

t = np.linspace(0, 10, 2**n)
t1 = np.linspace(0, 10*2, 2**(n+1) - 1)
x = np.sin(2*np.pi*t)
h = np.exp(-3*t)

X = fft(x.astype(complex))
H = fft(h.astype(complex))

xx = ifft(X)
print(x)
print(xx)
Y = X*H

yyy = ifft(Y)
print(yyy)

yyy = yyy.astype(float)

y = convolve(x, h)
yy = signal.convolve(x, h, method="fft")

plt.plot(t, x, label="1")
plt.plot(t, h, label="2")
plt.plot(t, y[0:2**n], label="3")
plt.plot(t, yy[0:2**n])
plt.plot(t, yyy/1000)

plt.grid()
plt.legend()
plt.show()

