import numpy as np
import matplotlib.pyplot as plt
import math


# 𝛤 = 10
# 𝑇_𝑚𝑎𝑥 = 20
# 𝑟_𝑡 = np.linspace(0, 50, 10000)  # Generate 100 equally spaced points between 0 and 50
#
# 𝑦 = np.where(𝑟_𝑡 <= 𝑇_𝑚𝑎𝑥, np.exp(𝛤 * (𝑟_𝑡 - 𝑇_𝑚𝑎𝑥) / 𝑇_𝑚𝑎𝑥), 1)


# cost 2
# B = 10
# t_max = 20
# target = t_max + 2*math.log(0.9)
#
# r = np.linspace(0, 50, 10000)  # Generate 100 equally spaced points between 0 and 50
#
# y = np.where(r <= target, np.exp(B * (r - t_max) / t_max), 0.9 + ((r - target) / (50 - target)) * 0.1)


# cost 3


# c_delay---------------------
t_max = 20
T_upper = 50
B = np.log(1+0.5)/((T_upper-t_max)/t_max)


r = np.linspace(0, 50, 10000)  # create response time data , 10000 point in (50, 10000)
y = np.where(r <= t_max, 0, np.exp(B*(r - t_max) / t_max)-0.5)


plt.plot(r, y)
plt.xlabel('r')
plt.ylabel('c_delay')
plt.yticks(np.arange(0., 1.1, 0.1))
plt.title('')
plt.grid(True)
plt.savefig("c_delay_function.png", dpi=300)
plt.show()

# c_res---------------------
k_values = np.arange(1, 4)  # k的值域 [1, 3]
c_values = np.linspace(0.8, 1, 100)  # c的值域 [0.8, 1]
x = np.outer(k_values, c_values)  # 计算x的值

y = x / 3  # 计算y的值

#
fig, ax = plt.subplots()
for i in range(len(k_values)):
    ax.plot(c_values, y[i], label=f'k={k_values[i]}')

ax.set_xlabel('c_res')
ax.set_ylabel('y')
ax.legend()
plt.grid(True)
plt.savefig("c_res_function.png", dpi=300)
plt.show()

# c_utilization---------------------
relative_cpu_utilization = np.linspace(0, 1, 10000)
x1 = 0.8
x2 = 1.0
y1 = t_max
y2 = T_upper
c_utilization = []
for util in relative_cpu_utilization:
    if util > 0.8:
        map_utilization = (util - x1) * ((y2 - y1) / (x2 - x1)) + t_max
        c_util = np.exp(B * (map_utilization - t_max) / t_max) - 0.5
    else:
        c_util = 0
    c_utilization.append(c_util)

plt.plot(relative_cpu_utilization, c_utilization)
plt.xlabel('r')
plt.ylabel('c_utilization')
plt.yticks(np.arange(0., 1.1, 0.1))
plt.title('')
plt.grid(True)
plt.savefig("c_utilization_function.png", dpi=300)
plt.show()


def map_function(util):
    x1 = 0.8
    x2 = 1.0
    y1 = t_max
    y2 = T_upper
    return (util - x1) * ((y2 - y1) / (x2 - x1)) + t_max

x = np.linspace(0.8, 1.0, 100)

y = map_function(x)


plt.plot(x, y)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Mapping Function')
plt.grid(True)
plt.show()


# test -------------------
t_max = 20
T_upper = 50
B = np.log(1+0.5)/((T_upper-t_max)/t_max)
relative_cpu_utilization = 0.9
if relative_cpu_utilization > 0.8:
    x1 = 0.8
    x2 = 1.0
    y1 = t_max
    y2 = T_upper

    clip_relative_cpu_utilization = min(relative_cpu_utilization, 1)
    map_utilization = (clip_relative_cpu_utilization - x1) * ((y2 - y1) / (x2 - x1)) + t_max
    c_utilization = np.exp(B * (map_utilization - t_max) / t_max) - 0.5
else:
    c_utilization = 0

print(c_utilization)
# test -------------------