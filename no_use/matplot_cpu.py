import matplotlib.pyplot as plt
import statistics

# delay modify = average every x delay (x = 10, 50, 100)
# request rate r
# r = '100'
simulation_time = 100  # s

# moving for plot
moving_avg = 0
move = 10

# limit_cpus = 1
# tmp_str = "result2/result_cpu" # result_1016/tm1
tmp_dir = "0520/request_40/result1/"
path1 = tmp_dir + "app_mn1_cpu.txt"
path2 = tmp_dir + "app_mn2_cpu.txt"
setting = tmp_dir + "setting.txt"
f = open(setting, "r")
cpus1 = 0
replica1 = 0
for line in f:
    s = line.split(': ')
    if s[0] == 'cpus':
        cpus1 = float(s[1])
    if s[0] == 'replica':
        replica1 = float(s[1])

Resource_use = cpus1 * replica1
service = ["First_level_mn1", "Second_level_mn2", "app_mnae1", "app_mnae2"]

# path_list = [path1]
path_list = [path1, path2]

def cal_cpu(f):
    cpu = []
    time = []

    for line in f:
        s = line.split(' ')
        # if float(s[2]) > 0 :
        #     # print(float(s[0]), float(s[2]))
        if float(s[0]) < simulation_time:
            time.append(float(s[0]))
            u = float(s[1])/cpus1
            if u > 100 : u = 100
            cpu.append(u)
    f.close()

    avg = sum(cpu) / len(cpu)
    max_d = max(cpu)
    min_d = min(cpu)
    st_dev = statistics.pstdev(cpu)
    print(avg, max_d, min_d)
    print("st_dev: ", st_dev)

    y = cpu
    #
    # count = 0
    # for i in range(simulation_time):
    #     r = time.count(i)
    #     if r > 0:
    #         d = 1 / r
    #         for j in range(r):
    #             x.append(count)
    #             count += d
    #     else:
    #         count += 1

    # print(len(time), len(cpu))

    if moving_avg:
        cpu_m = []
        for i in range(len(cpu)):
            if i < move:

                avg = sum(cpu[:i + 1]) / (i + 1)
            else:
                avg = sum(cpu[i - move + 1:i + 1]) / move

            cpu_m.append(avg)
        y = cpu_m

    data_dict = {}
    for index, value in zip(time, y):
        if index in data_dict:
            data_dict[index].append(value)
        else:
            data_dict[index] = [value]

    x_ = []
    y_ = []
    for index, values in data_dict.items():
        x_.append(index)
        y_.append(sum(values) / len(values))

    # print(x_)
    # print(y_)

    return x_, y_

# Plot --------------------------------------

def fig_add(x, y, label):
    #plt.subplot(pos)
    plt.plot(x, y, label=label)  # color=color

# pos = 141
tmp_count = 0
for p in path_list:

    f = open(p, "r")
    x, y = cal_cpu(f)
    # x_ = [i+1 for i in x]
    # print(y)

    ### plot delay
    fig_add(x, y, service[tmp_count])
    tmp_count += 1
    # fig_add(x, y1, 'Machine2', 'blue')
    # fig_add(x, y2, 'Machine2', 'blue')
    # fig_add(x, y3, 'Machine2', 'blue')
    # fig_add(x, y4, 'Machine2', 'blue')


# plt.title("Test")

plt.xlabel("timestamp")
plt.ylabel("Cpu utilization(%) ")
plt.grid(True)
plt.legend()
plt.xlim(0, simulation_time)
plt.ylim(0, 105)
plt.savefig(tmp_dir + "Cpu_utilization.png", dpi=300)
plt.tight_layout()
plt.show()