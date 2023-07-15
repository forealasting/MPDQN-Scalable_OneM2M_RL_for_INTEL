import matplotlib.pyplot as plt
import statistics

# r = 50
# data_name = '_tm1'
# data_name = str(r)
simulation_time = 100  # s
data_rate = 60

# moving for plot
moving_avg = 0  # choose avg response_time
move = 10

# timestamp average for plot
if_timestamp_average = 1

tmp_dir = "0520/request_70/result1/"
path1 = tmp_dir + "app_mn1_response.txt"
path2 = tmp_dir + "app_mn2_response.txt"
path_list = [path1, path2]
service = ["First_level_mn1", "Second_level_mn2", "app_mnae1", "app_mnae2"]

# ----------------for same data rate and different resource use --------#
# tmp_dir = "0519/request_40/"
# path1 = tmp_dir + "result1/app_mn1_response.txt"
# path2 = tmp_dir + "result2/app_mn1_response.txt"
# path3 = tmp_dir + "result3/app_mn1_response.txt"
# path4 = tmp_dir + "result4/app_mn1_response.txt"
# path5 = tmp_dir + "result5/app_mn1_response.txt"
# path6 = tmp_dir + "result6/app_mn1_response.txt"
# # path7 = tmp_dir + "result7/app_mn1_response.txt"
# path_list = [path1, path2, path3, path4, path5, path6]
# # path_list = [path1, path2, path3, path4, path5, path6, path7]
# service = ["cpus 0.5", "cpus 0.6", "cpus 0.7", "cpus 0.8", "cpus 0.9", "cpus 1.0"]


# # ----------------for different data rate and same resource use --------#
# title = "Resource use = 1.0 "
# tmp_dir = "0519/"
# tmp_dir1 = "0519/request_40/result6/"
# tmp_dir2 = "0519/request_50/result6/"
# tmp_dir3 = "0519/request_60/result6/"
# tmp_dir4 = "0519/request_70/result6/"
#
# path1 = tmp_dir1 + "app_mn1_response.txt"
# path2 = tmp_dir2 + "app_mn1_response.txt"
# path3 = tmp_dir3 + "app_mn1_response.txt"
# path4 = tmp_dir4 + "app_mn1_response.txt"
#
# path_list = [path1, path2, path3, path4]
# # path_list = [path1, path2, path3]
# service = ["data rate 40", "data rate 50", "data rate 60", "data rate 70"]


def cal_response_time(f, simulation_time, service_name):
    time = []
    response = []
    response_time = []

    for line in f:
        s = line.split(' ')
        try:
            time.append(float(s[0])+1)
            response.append(str(s[1]))
            if str(s[1]) == 'timeout' or float(s[2])>0.05:
                # print('timeout')
                tmp = 0.05
            else:
                tmp = float(s[2])
            response_time.append(tmp * 1000)

        except:
            print("error")
    f.close()

    x = []
    y = response_time
    count = 0
    for i in range(simulation_time):
        r = time.count(i)
        if r > 0:
            d = 1 / r
            for j in range(r):
                x.append(count)
                count += d
        else:
            count += 1


    if if_timestamp_average:
        tmp_count = 0
        avg_response_time = []
        x = []
        for i in range(simulation_time):
            r = time.count(i)
            if r > 0:
                avg_response_time.append(sum(response_time[tmp_count:tmp_count + r]) / r)
                x.append(i)
            tmp_count += r

        y = avg_response_time

    avg = sum(y) / len(y)
    max_d = max(y)
    min_d = min(y)
    st_dev = statistics.pstdev(y)
    loss_count = response.count('500')
    loss_rate = loss_count/len(response)
    print("avg: ", avg, "max: ", max_d, "min: ", min_d)
    print("st_dev: ", st_dev)


    return x, y


def fig_add(x, y, label):
    plt.plot(x, y, label=label)


tmp_count = 0
for p in path_list:

    f = open(p, "r")
    x, y = cal_response_time(f, simulation_time, p)
    ### plot # service[tmp_count] for show service name
    if tmp_count == 0:
        Rmax1 = 15
        Rmax2 = 20
        Rmax3 = 25
    else:
        Rmax1 = 15
        Rmax2 = 20
        Rmax3 = 25

    result1 = filter(lambda v: v > Rmax1, y)
    R1 = len(list(result1)) / len(y)
    result2 = filter(lambda v: v > Rmax2, y)
    R2 = len(list(result2)) / len(y)
    result3 = filter(lambda v: v > Rmax3, y)
    R3 = len(list(result3)) / len(y)
    print("Rmax1 violation: ", R1*100)
    print("Rmax2 violation: ", R2*100)
    print("Rmax3 violation: ", R3*100)
    fig_add(x, y, service[tmp_count])
    tmp_count += 1

# plt.title("Test")
plt.ylim(0, 100)
plt.fill()
# plt.title(title)
plt.xlabel("timestamp")
plt.ylabel("Responese time(ms)")
plt.grid(True)
plt.legend()

plt.savefig(tmp_dir + "Responese_time.png", dpi=300)
plt.show()