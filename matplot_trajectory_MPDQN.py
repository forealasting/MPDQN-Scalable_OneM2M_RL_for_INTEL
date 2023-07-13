import matplotlib.pyplot as plt
import statistics
import re
import json
import warnings
import os
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)
# delay modify = average every x delay (x = 10, 50, 100)
# request rate r
# r = '100'

total_episodes = 16
step_per_episodes = 60

# evaluation
if_evaluation = 1
if if_evaluation:
    total_episodes = 1
# tmp_str = "result2/result_cpu" # result_1016/tm1
#tmp_dir = "pdqn_result/result2"
# tmp_dir = "offline/database4"
tmp_dir = "mpdqn_result/result10/evaluate/"
path1 = tmp_dir + "/app_mn1_trajectory.txt"
path2 = tmp_dir + "/app_mn2_trajectory.txt"

service = ["First_level_MNCSE", "Second_level_MNCSE", "app_mnae1", "app_mnae2"]
Rmax_mn1 = 20
Rmax_mn2 = 20
# path_evaluate = tmp_dir+"/evaluate/"
# if not os.path.exists(path_evaluate):
#     os.makedirs(path_evaluate)
# service = ["Second_level_mn1", "First_level_mn1", "app_mnae1", "app_mnae2"]
# path_list = [path1, path2]
path_list = [path1, path2]
def parse(p):
    with open(p, "r") as f:
        data = f.read().splitlines()
        parsed_data = []
        parsed_line = []
        tmp = 0
        for line in data:
            tmp += 1
            # parse data
            # match = re.match(
            #     r"(\d+) \[(.+)\] (\d+) ([-+]?\d*\.\d+) ([-+]?\d*\.\d+) ([-+]?\d*\.\d+) ([-+]?\d*\.\d+) \[(.+)\] (\w+)",
            #     line)  # for PDQN/MPDQN
            match = re.match(
                r"(\d+) \[(.+)\] (\d+) \[(.+)\] ([-+]?\d*\.\d+) ([-+]?\d*\.\d+) ([-+]?\d*\.\d+) \[(.+)\] (\w+)", line)

            # assert False

            if match != None:
                # Convert the parsing result to the corresponding Python object
                # line_data = [int(match.group(1)), json.loads("[" + match.group(2) + "]"), int(match.group(3)),
                #              float(match.group(4)), float(match.group(5)), float(match.group(6)),
                #              json.loads("[" + match.group(7) + "]"), match.group(8) == "True"]  # for DQN/Qlearning
                # line_data = [int(match.group(1)), json.loads("[" + match.group(2) + "]"), int(match.group(3)),
                #              float(match.group(4)), json.loads("[" + match.group(5) + "]"), match.group(6) == "True"]

                # line_data = [int(match.group(1)), json.loads("[" + match.group(2) + "]"), int(match.group(3)),
                #              float(match.group(4)), float(match.group(5)), float(match.group(6)), float(match.group(7)),
                #              json.loads("[" + match.group(8) + "]"), match.group(9) == "True"]  # for PDQN/MPDQN
                line_data = [int(match.group(1)), json.loads("[" + match.group(2) + "]"), int(match.group(3)),
                             json.loads("[" + match.group(4) + "]"), float(match.group(5)), float(match.group(6)),
                             float(match.group(7)), json.loads("[" + match.group(8) + "]"), match.group(9) == "True"]
                parsed_line.append(line_data)
                # tmp == step_per_episodes
                if match.group(9) == "True":
                    parsed_data.append(parsed_line)
                    tmp = 0
                    parsed_line = []

    return parsed_data
    # return step, replica, cpu_utilization, cpus, reward, resource_use


# Plot --------------------------------------

def fig_add_Cpus(x, y, service_name):
    plt.figure()
    plt.plot(x, y, color="blue")  # color=color
    plt.title(service_name)
    #　plt.xlabel("step")
    plt.xlabel("step")
    plt.ylabel("Cpus")
    # plt.grid(True)

    plt.xlim(0, total_episodes*step_per_episodes)
    plt.ylim(0, 1.1)
    plt.savefig(tmp_dir + service_name + "_Cpus.png", dpi=300)
    plt.tight_layout()
    plt.show()


def fig_add_Replicas(x, y, service_name):
    plt.figure()
    plt.plot(x, y, color="green")  # color=color
    plt.title(service_name)
    plt.xlabel("step")
    plt.ylabel("Replicas")
    # plt.grid(True)

    plt.xlim(0, total_episodes*step_per_episodes)
    plt.ylim(0, 4)
    plt.savefig(tmp_dir + service_name + "_Replicas.png", dpi=300)
    plt.tight_layout()
    plt.show()


def fig_add_Cpu_utilization(x, y, y_, service_name):
    if not if_evaluation:
        plt.plot(x, y, color='royalblue', alpha=0.2)  # color=color # label=label
        plt.plot(x, y_, color='royalblue')  # color=color
    else:
        plt.plot(x, y, color='royalblue')  # color=color # label=label

    avg = sum(y) / len(y)
    plt.title(service_name + " Avg : " + str(avg))
    plt.xlabel("step")
    plt.ylabel("Cpu_utilization")
    # plt.grid(True)
    plt.xlim(0, total_episodes*step_per_episodes)
    plt.ylim(0, 110)
    plt.savefig(tmp_dir + service_name + "_Cpu_utilization.png", dpi=300)
    plt.tight_layout()
    plt.show()


def fig_add_response_times(x, y, y_, service_name):
    plt.figure()
    if not if_evaluation:
        plt.plot(x, y, color="purple", alpha=0.2)  # color=color # label=label
        plt.plot(x, y_, color="purple")  # color=color # label=label
    else:
        plt.plot(x, y, color="purple")  # color=color # label=label
    avg = sum(y) / len(y)

    plt.title(service_name + " Avg : " + str(avg))
    plt.xlabel("step")
    plt.ylabel("Response time")
    if service_name == "First_level_mn1":
        Rmax = Rmax_mn1
    else:
        Rmax = Rmax_mn2

    result2 = filter(lambda v: v > Rmax, y)
    R = len(list(result2)) / len(y)
    print("Rmax violation: ", R)

    # plt.grid(True)
    plt.axhline(y=Rmax_mn1, color='r', linestyle='--')
    plt.xlim(0, total_episodes*step_per_episodes)
    plt.ylim(0, 100)
    plt.savefig(tmp_dir + service_name + "_Response_time.png", dpi=300)
    plt.tight_layout()
    plt.show()


def fig_add_Resource_use(x, y, y_, service_name, dir):
    plt.figure()
    x = [i for i in range(len(y))]

    if not if_evaluation:
        plt.plot(x, y, color="black", alpha=0.2)  # color=color # label=label
        plt.plot(x, y_, color="black")  # color=color
    else:
        plt.plot(x, y, color="black")  # color=color # label=label
    # print(len(y))

    avg = sum(y) / len(y)
    avg = round(avg, 2)
    print(service_name + " Avg_Resource_use", avg)

    plt.title(service_name + " Avg : " + str(avg))
    plt.xlabel("step")
    plt.ylabel("Resource_use")
    # plt.grid(True)
    plt.xlim(0, total_episodes*step_per_episodes)
    plt.ylim(0, 3)
    plt.savefig(dir + service_name + "_Resource_use.png", dpi=300)
    plt.tight_layout()
    plt.show()

def fig_add_reward(x, y, y_, service_name):
    if if_evaluation:
        x = x[:-1]
    plt.figure()
    plt.plot(x, y, color="red", alpha=0.2)  # color=color # label=label
    plt.plot(x, y_, color="red")  # color=color # label=label
    avg = sum(y) / len(y)
    plt.title(service_name + " Avg : " + str(avg))
    plt.xlabel("step")
    plt.ylabel("Reward")

    # plt.grid(True)

    plt.xlim(0, total_episodes*step_per_episodes)
    plt.ylim(-0.6, 0)
    plt.savefig(tmp_dir + service_name + "_cost.png", dpi=300)
    plt.tight_layout()
    plt.show()

def moving_average(lst, move=5):
    moving_averages = []
    for i in range(len(lst)):
        start_idx = max(0, i - move)
        end_idx = min(i + move + 1, len(lst))
        window = lst[start_idx:end_idx]
        average = sum(window) / len(window)
        moving_averages.append(average)

    return moving_averages


def parse_episods_data(episods_data, service_name):
    plot_name = ["replica", "cpu_utilization", "cpus", "reward", "resource_use"]
    tmp_step = 0
    step = []
    replicas = []
    cpu_utilization = []
    cpus = []
    reward = []
    response_times = []

    for episode in range(1, total_episodes+1):
        for parsed_line in episods_data[episode-1]:
            # parsed_line = episods_data[episode-1]
            # step.append(parsed_line[0])
            step.append(tmp_step)
            replicas.append(parsed_line[1][0])
            cpus.append(parsed_line[1][2])
            cpu_u = parsed_line[1][1]*100
            if cpu_u > 100:
                cpu_u = 100
            cpu_utilization.append(cpu_u)
            response_times.append(parsed_line[1][3])
            reward.append(parsed_line[4])  # cost = -reward
            tmp_step += 1
            if tmp_step == step_per_episodes and if_evaluation:
                step.append(tmp_step)
                replicas.append(parsed_line[7][0])
                cpus.append(parsed_line[7][2])
                cpu_u = parsed_line[7][1] * 100
                if cpu_u > 100:
                    cpu_u = 100
                cpu_utilization.append(cpu_u)
                response_times.append(parsed_line[7][3])
        # episode_reward.append(sum(reward)/len(reward))
    resource_use = [x * y for x, y in zip(replicas, cpus)]
    replicas_ = moving_average(replicas)
    response_times_ = moving_average(response_times)
    cpu_utilization_ = moving_average(cpu_utilization)
    cpus_ = moving_average(cpus)
    reward_ = moving_average(reward)
    resource_use_ = moving_average(resource_use)
    fig_add_Cpus(step, cpus, service_name)
    fig_add_Replicas(step, replicas, service_name)
    fig_add_response_times(step, response_times, response_times_, service_name)
    fig_add_Cpu_utilization(step, cpu_utilization, cpu_utilization_, service_name)
    fig_add_Resource_use(step, resource_use, resource_use_, service_name, tmp_dir)
    fig_add_reward(step, reward, reward_, service_name)


tmp_count = 0
for p in path_list:

    episods_data = parse(p)
    # print(len(episods_data), episods_data)
    # step, replica, cpu_utilization, cpus, reward, resource_use = parse_episods_data(episods_data)
    # print(len(episods_data))
    parse_episods_data(episods_data, service[tmp_count])

    tmp_count += 1

#
#
# path_1 = tmp_dir + "/app_mn1_response.txt"
# path_2 = tmp_dir + "/app_mn2_response.txt"
# path_list = [path_1, path_2]
#
# service = ["First_level_mn1", "Second_level_mn2", "app_mnae1", "app_mnae2"]
# # service = ["First_level_mn1", "Second_level_mn2", "app_mnae1", "app_mnae2"]
# tmp = 0
# for path in path_list:
#     with open(path, "r") as f:
#         data = f.read().splitlines()
#         response_time = []
#         for line in data:
#             rt = float(line.split()[2])
#             if rt > 0.05:
#                 rt = 0.05
#             response_time.append(rt)
#         #print(response_time)
#         x = []
#         y = []
#         tmp1 = 0
#         for i in range(0, len(response_time), 5):
#             rt_sum = 0
#             for j in range(i, i + 5):
#                 rt_sum += float(response_time[j])
#             y.append(rt_sum / 5 * 1000)
#             x.append(tmp1)
#             tmp1 += 1
#
#         y_m = moving_average(y)
#         plt.figure()
#
#         # plt.plot(x, y_m, color="purple", alpha=0.2)  # color=color # label=label
#         plt.plot(x, y_m, color="purple")  # color=color # label=label
#         avg = sum(y) / len(y)
#         if service[tmp] == "First_level_mn1":
#             Rmax = Rmax_mn1
#         else:
#             Rmax = Rmax_mn2
#
#
#         result2 = filter(lambda v: v > Rmax, y_m)
#         R = len(list(result2)) / len(y_m)
#         print("Rmax violation: ", R)
#         avg_m = sum(y_m) / len(y_m)
#         print(service[tmp] + "avg_m: ", avg_m)
#         print(service[tmp] + "avg: ", avg)
#         avg = sum(y) / len(y)
#         plt.title(service[tmp] + " Avg : " + str(avg))
#         plt.xlabel("step")
#         plt.ylabel("Response")
#         # plt.grid(True)
#         plt.xlim(0, total_episodes*60)
#         plt.axhline(y=Rmax, color='r', linestyle='--')
#         plt.ylim(0, 100)
#         plt.savefig(tmp_dir + service[tmp] + "_Response.png")
#         plt.tight_layout()
#         plt.show()
#     tmp+=1






