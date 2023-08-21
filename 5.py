import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# # PART A

# x_1 = np.linspace(-2.5, 2.5, 1000)
# y_1 = x_1**2 - 2

# y_2 = np.linspace(-2.5, 2.5, 1000)
# x_2 = y_2**2 - 2

# plt.plot(x_1, y_1, label='y = x^2 - 2')
# plt.plot(x_2, y_2, label='x = y^2 - 2')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis('scaled')
# plt.xlim(-2.5, 2.5)
# plt.ylim(-2.5, 2.5)
# plt.legend()
# plt.show()

# PART C

def f_1(x, y):
    return x**2 - y - 2

def f_2(x, y):
    return y**2 - x - 2

def del_x_f_1(x, y):
    return 2 * x

def del_y_f_1(x, y):
    return -1

def del_x_f_2(x, y):
    return -1

def del_y_f_2(x, y):
    return 2 * y

x_0 = 0.5
y_0 = 0.5

x_k = x_0
y_k = y_0
counter = 0

while counter < 5:
    J = np.array([[del_x_f_1(x_k, y_k), del_y_f_1(x_k, y_k)],
                  [del_x_f_2(x_k, y_k), del_y_f_2(x_k, y_k)]])
    
    f = np.array([[f_1(x_k, y_k)],
                 [f_2(x_k, y_k)]])
    
    delta_k = np.linalg.solve(J, -f)
    
    x_k += delta_k[0, 0]
    y_k += delta_k[1, 0]
    
    counter += 1

    print(x_k, y_k)