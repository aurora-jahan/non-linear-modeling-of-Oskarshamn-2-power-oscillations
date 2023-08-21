import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sys import exit

# PART A

file_name = "Relaxation_modified.txt"

r = []
n = []

with open(file_name, "r") as f:
    data_list = [line.rstrip("\n") for line in f]

for data in data_list:
    r.append(float(data.split(' ')[0]))
    n.append(float(data.split(' ')[1]))

r = np.asarray(r)
n = np.asarray(n)
y = np.log(r**2 * n)

plt.plot(r, y, color='blue', marker='o', label='experimental data')
# plt.plot(x, y_hat, color='red', label='regression line')
# plt.plot(x, y_hat_2, color='green')
plt.xlabel('distance (cm)')
plt.ylabel('y = ln[r^2 n(r)]')
plt.legend()
plt.show()
# exit()

# PART C

# SCIPY LINEAR REGRESSION

result = sp.stats.linregress(r, y)

print(result.intercept, result.slope, result.rvalue)

# PART D

print(- 1 / result.slope)

# PART E

y_hat = result.slope * r + result.intercept

plt.plot(r, y, color='blue', marker='o', label='experimental data')
plt.plot(r, y_hat, color='red', label='regression line')
# plt.plot(x, y_hat_2, color='green')
plt.xlabel('distance (cm)')
plt.ylabel('y = ln[r^2 n(r)]')
plt.legend()
plt.show()

# PART F

RMSE = np.sqrt(np.average((y_hat - y)**2))
RMSN = RMSE / np.std(y)
CoD = 1 - RMSN**2
pearson_corr_coeff = np.sqrt(CoD)

print(pearson_corr_coeff, RMSE, RMSN, CoD)