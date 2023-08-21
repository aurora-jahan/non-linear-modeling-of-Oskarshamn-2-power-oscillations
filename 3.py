import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

file_name = "CTE_Steel.txt"

x = []
y = []

with open(file_name, "r") as f:
    data_list = [line.rstrip("\n") for line in f]

for data in data_list:
    x.append(float(data.split(' ')[0]))
    y.append(float(data.split(' ')[1]))

x = np.asarray(x)
y = np.asarray(y)

# PART A

A = np.array([[1, np.average(x)],
             [np.average(x), np.average(x**2)]])

A_0 = np.array([[np.average(y), np.average(x)],
             [np.average(x * y), np.average(x**2)]])

A_1 = np.array([[1, np.average(y)],
             [np.average(x), np.average(x * y)]])

alpha_0 = np.linalg.det(A_0) / np.linalg.det(A)
alpha_1 = np.linalg.det(A_1) / np.linalg.det(A)

print(alpha_0, alpha_1)

# alpha_1_2 = np.cov(x, y)[0, 1] / np.var(x)
# alpha_0_2 = np.average(y) - np.average(x) * alpha_1_2

# print(alpha_0_2, alpha_1_2)

# PART B

y_hat = alpha_1 * x + alpha_0
# y_hat_2 = alpha_1_2 * x + alpha_0_2

plt.plot(x, y, color='blue', marker='o', label='experimental data')
plt.plot(x, y_hat, color='red', label='regression line')
# plt.plot(x, y_hat_2, color='green')
plt.xlabel('T (C)')
plt.ylabel('CTE (* 10^(-6)  1/C)')
plt.legend()
# plt.show()

# PART C

RMSE = np.sqrt(np.average((y_hat - y)**2))
RMSN = RMSE / np.std(y)
CoD = 1 - RMSN**2
pearson_corr_coeff = np.sqrt(CoD)

print(pearson_corr_coeff, RMSE, RMSN, CoD)

# # NUMPY LINEAR REGRESSION

# A = np.vstack([x, np.ones(len(x))]).T
# m, c = np.linalg.lstsq(A, y, rcond=None)[0]

# print(c, m)

# # SCIPY LINEAR REGRESSION

# result = sp.stats.linregress(x, y)

# print(result.intercept, result.slope, result.rvalue)