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

t = np.asarray(t)
power_perc = np.asarray(power_perc)

avg_power = np.average(power_perc)
std_power = np.std(power_perc)

fig, (top, bottom) = plt.subplots(2)
fig.suptitle('Reactor power oscillations')

top.plot(t, power_perc, color='blue', label='experimental data')
top.hlines(avg_power, t[0], t[-1], color='red', label='mean value line')
# plt.plot(x, y_hat, color='green', label='regression line')
top.set_xlabel('time (s)')
top.set_ylabel('Power (%)')
top.set_title(f'Unprocessed data; Mean = {round(avg_power, 3)} %, STD = {round(std_power, 3)} %')
top.legend()

# PART B

preprocessed_power_perc = sp.signal.detrend(power_perc, type='linear')

# PART C

std_1 = np.std(preprocessed_power_perc)

preprocessed_power_perc = preprocessed_power_perc * std_1

std_2 = np.std(preprocessed_power_perc)

# PART D

bottom.plot(t, preprocessed_power_perc, color='blue', label='experimental data')
bottom.hlines(0, t[0], t[-1], color='red', label='t axis')
# plt.plot(x, y_hat, color='green', label='regression line')
bottom.set_xlabel('time (s)')
bottom.set_ylabel('Power (%)')
bottom.set_title(f'Preprocessed data; STD before scaling = {round(std_1, 3)} %, STD after scaling = {round(std_2, 3)} %')
bottom.legend()

plt.show()
# exit()