import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# PART A

file_name = "power_oscillations_data.txt"

t = []
power_perc = []

with open(file_name, "r") as f:
    data_list = [line.rstrip("\n") for line in f]

for data in data_list:
    t.append(float(data.split('   ')[0]))
    power_perc.append(float(data.split('   ')[1]))

t = np.asarray(t) - t[0]
power_perc = np.asarray(power_perc)

preprocessed_power_perc = sp.signal.detrend(power_perc, type='linear')

std_1 = np.std(preprocessed_power_perc)

preprocessed_power_perc = preprocessed_power_perc * std_1

# PART B

t_1 = t[130 + np.argmax(preprocessed_power_perc[130:150])]
t_2 = t[150 + np.argmax(preprocessed_power_perc[150:170])]
t_3 = t[170 + np.argmax(preprocessed_power_perc[170:190])]

peak_1 = preprocessed_power_perc[130 + np.argmax(preprocessed_power_perc[130:150])]
peak_2 = preprocessed_power_perc[150 + np.argmax(preprocessed_power_perc[150:170])]
peak_3 = preprocessed_power_perc[170 + np.argmax(preprocessed_power_perc[170:190])]

plt.plot(t, preprocessed_power_perc, color='b', marker='o', label='experimental data')
plt.hlines(0, t[0], t[-1], color='red', label='t axis')
# plt.plot(x, y_hat, color='green', label='regression line')
plt.xlabel('time (s)')
plt.ylabel('Power (%)')
plt.title(f'Preprocessed data')
plt.grid(visible=True)
plt.legend()
plt.show()

T = (t_3 - t_1) / 2
FR = 1/T
omega = 2 * np.pi / T

print(f'T = {round(T, 3)} sec, FR = {round(FR, 3)} per sec, omega = {round(omega, 3)} radians per sec')

# PART C

print(f'peak_1 = {round(peak_1, 3)}, peak_2 = {round(peak_2, 3)}, peak_3 = {round(peak_3, 3)}')

DR_1 = peak_2 / peak_1
DR_2 = peak_3 / peak_2
DR_3 = (DR_1 + DR_2) / 2

print(f'DR_1 = {round(DR_1, 3)}, DR_2 = {round(DR_2, 3)}, DR_3 = {round(DR_3, 3)}')

# PART D

Lambda = - FR * np.log(DR_3)

a_1 = peak_1 * np.exp(Lambda * t_1)
a_2 = peak_2 * np.exp(Lambda * t_2)
a_3 = peak_3 * np.exp(Lambda * t_3)

a = (a_1 + a_2 + a_3) / 3

print(f'Lambda = {round(Lambda, 3)}, a = {round(a, 3)}')

# PART E

phi_1 = 2 * 7 * np.pi - omega * t_1
phi_2 = 2 * 8 * np.pi - omega * t_2

phi = (phi_1 + phi_2) / 2

print(f'phi = {round(phi, 3)}')
