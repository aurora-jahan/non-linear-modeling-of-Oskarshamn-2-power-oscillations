import numpy as np
from numpy import exp, cos
import scipy as sp
from scipy.optimize import leastsq, curve_fit, least_squares
import matplotlib.pyplot as plt

# PART 0

file_name = "power_oscillations_data.txt"

t = []
power_perc = []

with open(file_name, "r") as f:
    data_list = [line.rstrip("\n") for line in f]

for data in data_list:
    t.append(float(data.split('   ')[0]))
    power_perc.append(float(data.split('   ')[1]))

t_1 = np.asarray(t)
t_2 = t_1 - t[0]

raw_data = np.asarray(power_perc)

data_1 = sp.signal.detrend(power_perc, type='linear')

std_1 = np.std(data_1)

data_2 = data_1 * std_1

alpha_0 = [15.472, -0.144, 2.992, 1.197]

# PART 1A

# # scipy.optimize.leastsq (LEGACY wrapper)

# def residual(alpha, t, data):
#     """Model a decaying cosine wave and subtract data."""
#     a = alpha[0]
#     Lambda = alpha[1]
#     omega = alpha[2]
#     phi = alpha[3]

#     model = a * exp(- Lambda * t) * cos(omega * t + phi)

#     return data - model

# out = leastsq(residual, alpha, args=(t_2, data_2))
# print(out)    # (array([15.58683264, -0.13119207,  3.04356232,  0.35896956]), 1)


# # scipy.optimize.curvefit

# def model(t, a, Lambda, omega, phi):
#     model = a * exp(- Lambda * t) * cos(omega * t + phi)
#     return model

# popt, pcov = curve_fit(f=model, xdata=t_2, ydata=data_2, p0=alpha_0)

# print(popt) # [15.5868328  -0.13119207  3.04356231  0.35896969]

# scipy.optimize.least_squares

def residual(alpha, t, data):
    a = alpha[0]
    Lambda = alpha[1]
    omega = alpha[2]
    phi = alpha[3]
    
    model = a * exp(- Lambda * t) * cos(omega * t + phi)

    return data - model
    
res_lsq = least_squares(residual, alpha_0, xtol=1e-6, args=(t_2, data_2), verbose=0)

# print(res_lsq.nfev)

# PART 1B

# print(res_lsq.x)    # [15.58683292 -0.13119207  3.04356232  0.35896964]

DR = exp(- 2 * np.pi * res_lsq.x[1] / res_lsq.x[2])

# print(DR)

def model(alpha, t):
    a = alpha[0]
    Lambda = alpha[1]
    omega = alpha[2]
    phi = alpha[3]
    
    model = a * exp(- Lambda * t) * cos(omega * t + phi)

    return model

y_hat = model(res_lsq.x, t_2)

RMSE = np.sqrt(np.average((y_hat - data_2)**2))
RMSN = RMSE / np.std(data_2)
CoD = 1 - RMSN**2
pearson_corr_coeff = np.sqrt(CoD)

# print(pearson_corr_coeff, RMSE, RMSN, CoD)

# Part 1C

plt.plot(t_2, data_2, 'b--', label='original signal (preprocessed)')
plt.plot(t_2, y_hat, 'r-', label='found model')
plt.xlabel('time (s)')
plt.ylabel('Power (%)')
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21])
plt.yticks([-200, -150, -100, -50, 0, 50, 100, 150, 200])
plt.title(f'Comparison between the experimental data and the fitted model')
plt.grid(visible=True)
plt.legend()
plt.show()