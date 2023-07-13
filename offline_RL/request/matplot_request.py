import matplotlib.pyplot as plt


# delay modify = average every x delay (x = 10, 50, 100)
# request rate r

simulation_time = 3602  # 300 s
use_tm = 1
r = 300


path1 = "request13.txt"

path_list = [path1]

def cal_req(f):
    # time = [x for x in range(simulation_time)]
    req = []

    for line in f:
        req.append(int(line))

    f.close()

    return req


# Plot --------------------------------------

def fig_add(x, y, label):
    plt.plot(x, y, label=label)


for p in path_list:

    f = open(p, "r")
    x = [k for k in range(simulation_time)]
    if use_tm:
        y = cal_req(f)
    else:
        y = []
        for i in range(simulation_time):
            y.append(r)
        # r_tmp = 10
        # for i in range(1, 31):
        #     for j in range(10):
        #         y.append(r_tmp)
        #     r_tmp += 10

    print(len(x), len(y))
    print(y)

    # print(y)

    ### plot delay
    fig_add(x, y, 'Machine1')


plt.title("Workload")
plt.xlabel("timestamp")
plt.ylabel("Data rate(requests/s) ")
plt.grid(True)
plt.ylim(0,110)
# plt.legend()
plt.savefig("Data_rate.png")
plt.show()


