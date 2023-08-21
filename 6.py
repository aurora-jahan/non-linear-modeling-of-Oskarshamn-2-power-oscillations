import numpy as np
import matplotlib.pyplot as plt

def f(t):
    a = 1
    Lambda = 0.2
    omega = 3
    phi = 4
    
    return a * np.exp(- Lambda * t) * np.cos(omega * t + phi)

t = np.linspace(-10, 10, 10000)

y = f(t)

def amp(t):
    a = 1
    Lambda = 0.2
    omega = 3
    phi = 4
    
    return a * np.exp(- Lambda * t)

amplitude = amp(t)

def t_m_published(n):
    a = 1
    Lambda = 0.2
    omega = 3
    phi = 4
    
    return (2 * n * np.pi - phi) / omega  # given everywhere but actually not the true maxima
    
    # return (1/omega) * (np.arctan(-Lambda/omega) + n * np.pi - phi)   # actual maxima
    

def t_m_actual(n):
    a = 1
    Lambda = 0.2
    omega = 3
    phi = 4
    
    return (1/omega) * (np.arctan(-Lambda/omega) + 2 * n * np.pi - phi)   # actual maxima

plt.plot(t, y)
plt.plot(t, amplitude)
plt.vlines([t_m_actual(-3), t_m_published(-3)], [0, 0], [5, 5])
plt.show()