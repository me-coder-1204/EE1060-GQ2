import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft
import cmath
import time

def convolve(f, h):
    
    X = np.zeros(len(f)+len(h)-1)
    for i in range(len(f)+len(h)-1):
        s = 0
        for j in range(len(f)):
            if i>=j and i < j + len(h):
                s += x[j] * h[i-j]
        X[i] = s
    return X

def fft_(x):
    n = len(x)
    if n==1:
        return x
    om = cmath.exp(-2*np.pi*1j/n)
    Pe = x[0::2]
    Po = x[1::2]
    ye, yo = fft_(Pe), fft_(Po)
    y = np.zeros(n).astype(complex)
    for i in range(int(n/2)):
        y[i] = ye[i] + om**i * yo[i]
        y[int(i + n/2)] = ye[i] - om**i * yo[i]
    return y

def _ifft_rec(x):
    n = len(x)
    if n==1:
        return x
    om = cmath.exp(2*np.pi*1j/n)
    Pe = x[0::2]
    Po = x[1::2]
    ye, yo = _ifft_rec(Pe), _ifft_rec(Po)
    y = np.zeros(n).astype(complex)
    for i in range(int(n/2)):
        y[i] = ye[i] + om**i * yo[i]
        y[int(i + n/2)] = ye[i] - om**i * yo[i]
    return y

def ifft_(X):
    return _ifft_rec(X).astype(float)/len(X)

def fft_d(x):
    n = len(x)
    if n==1:
        return np.array([-x[0], x[0]])
    om = cmath.exp(-np.pi*1j/n)
    Pe = x[0::2]
    Po = x[1::2]
    ye, yo = fft_d(Pe), fft_d(Po)
    y = np.zeros(2*n).astype(complex)
    for i in range(int(n)):
        y[i] = ye[i] + om**i * yo[i]
        y[int(i + n)] = ye[i] - om**i * yo[i]
    return y


n = 10
W = 10
t = np.linspace(-W, W, 2**n)
t1 = np.linspace(-2*W, 2*W, 2**(n+1) - 1)
x = np.sin(2*t)
#h = np.exp(-3*t)
T = 2
h = np.repeat(np.array([0,1,0]), [int(2**n*((W-T)/(2*W))),int(2**n*(T/W)),int(2**n*((W-T)/(2*W)))+2])
print(len(h))
print([2**n*int((W-T)/(2*W)),2**n*int(T/W),2**n*int((W-T)/(2*W))])
st = time.time()
y = convolve(x, h)
end = time.time()
print("Discreet Convolution: ", end-st)

st = time.time()
yy = signal.convolve(x, h)
end = time.time()
print("Scipy's Discreet Convolution", end-st)

X = fft_(x.astype(complex))
H = fft_(h.astype(complex))

st = time.time()
Xd = fft_d(x)
Hd = fft_d(h)
Yd = Xd*Hd
yd = ifft_(Yd)
end = time.time()
print("Discreen Convolution (FFT): ", end-st)

Y = X*H
yyy = ifft_(Y)
plt.plot(t, x, label=r"Input - x = sin(2t)")
plt.plot(t, h, label=r"Kernel - h")
#plt.plot(t, y[0:2**n], label="3")
plt.plot(t1, yd[1:], label="Convolution (with FFT)")
#plt.plot(t, yyy, label="5")
#plt.plot(t, yd[0:2**n], color="green", label="6")
plt.grid()
plt.legend()

plt.savefig("figs/FFT.png")
plt.show()

