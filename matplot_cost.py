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
# T_max = 20
# target = T_max + 2*math.log(0.9)
#
# r = np.linspace(0, 50, 10000)  # Generate 100 equally spaced points between 0 and 50
#
# y = np.where(r <= target, np.exp(B * (r - T_max) / T_max), 0.9 + ((r - target) / (50 - target)) * 0.1)


# cost 3
T_max = 20
r = np.linspace(0, 50, 10000)  # create response time data , 10000 point in (50, 10000)
y = np.where(r <= T_max, 0, np.exp(0.27*(r - T_max) / 20)-0.5)


plt.plot(r, y)
plt.xlabel('r')
plt.ylabel('c_perf')
plt.yticks(np.arange(0., 1.1, 0.1))
plt.title('')
plt.grid(True)
plt.savefig("cperf_function.png", dpi=300)
plt.show()


K_max = 3
c_max = 1
k_alloc = np.arange(1, 4)               # k的值域 [1, 3]
c_alloc = np.linspace(0.8, 1.0, 10000)  # c的值域 [0.8, 1] # create resource use data , 10000 point in (50, 10000)
x = np.outer(k_alloc, c_alloc)
y = x/(K_max*c_max)


fig, ax = plt.subplots()
for i in range(len(k_alloc)):
    ax.plot(c_alloc, y[i], label=f'k={k_alloc[i]}')

ax.set_xlabel('cpus allocation')
ax.set_ylabel('c_res')
ax.legend()
plt.savefig("cres_function.png", dpi=300)
plt.show()
# plt.plot(x, y)
# plt.xlabel('resouce use')
# plt.ylabel('cres')
# plt.yticks(np.arange(0., 1.1, 0.1))
# plt.title('')
# plt.grid(True)
# plt.savefig("cres_function.png", dpi=300)
# plt.show()


# r = 50
# y = np.where(r <= target, np.exp(B * (r - T_max) / T_max), 0.9 + ((r - target) / (50 - target)) * 0.1)
# print(r, y)