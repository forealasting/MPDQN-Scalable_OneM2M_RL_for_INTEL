import matplotlib.pyplot as plt


# delay modify = average every x delay (x = 10, 50, 100)
# request rate r
# r = '100'
episodes = 5  # 300 s

# limit_cpus = 1
# tmp_str = "result2/result_cpu" # result_1016/tm1
tmp_str = "all_result/result4_0215"
path1 = tmp_str + "/app_mn1_reward.txt"

service = ["app_mn1", "app_mn2", "app_mnae1", "app_mnae2"]

# path3 = tmp_str + str(limit_cpus) + "/tmp1/output_cpu100.txt"
# path4 = tmp_str + str(limit_cpus) + "/output_cputm1.txt"
# path = "output_cpu" + str(r) + ".txt"
# path_list = [path1]
path_list = [path1]

# Plot --------------------------------------

def fig_add(x, y, label):
    #plt.subplot(pos)
    plt.plot(x, y, label=label)  # color=color
    plt.xticks(x)  # set xticks


# pos = 141
tmp_count = 0
for p in path_list:

    f = open(p, "r")
    x = [i for i in range(1, episodes+1)]
    y = []
    for line in f:
        y.append(-float(line))  # - : reward to cost

    ### plot delay
    fig_add(x, y, service[tmp_count])
    tmp_count += 1
    # fig_add(x, y1, 'Machine2', 'blue')
    # fig_add(x, y2, 'Machine2', 'blue')
    # fig_add(x, y3, 'Machine2', 'blue')
    # fig_add(x, y4, 'Machine2', 'blue')



plt.title("Average cost of first level")
plt.xlabel("episode")
plt.ylabel("cost ")
plt.grid(True)
plt.legend()
plt.xlim(1, episodes)
plt.ylim(0, 1)
plt.savefig("cost.png")
plt.tight_layout()
plt.show()
