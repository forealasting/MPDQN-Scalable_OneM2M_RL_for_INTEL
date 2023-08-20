import matplotlib.pyplot as plt
import re
import json
import warnings
import os
from matplotlib import MatplotlibDeprecationWarning
import matplotlib.lines as mlines
import matplotlib
import statistics

warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)
# delay modify = average every x delay (x = 10, 50, 100)
# request rate r
# r = '100'
simulation_time = 3600  # 3600 s
total_episodes = 8
step_per_episodes = 30

# evaluation
if_evaluation = 1
if if_evaluation:
    total_episodes = 1

# tmp_str = "result2/result_cpu" # result_1016/tm1
# tmp_dir = "dqn_result/result1/evaluate/"
# tmp_dir = "dqn_result/result2/"
tmp_dir = "home/user/MPDQN-Scalable_OneM2M_RL/threshold_result/result2/"
path1 = tmp_dir + "/app_mn1_trajectory.txt"
path2 = tmp_dir + "/app_mn2_trajectory.txt"

service = ["First_level_MNCSE", "Second_level_MNCSE", "app_mnae1", "app_mnae2"]
Rmax_mn1 = 20
Rmax_mn2 = 5

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

        for line in data:
            # parse data
            match = re.match(r"(\d+) \[(.+)\] (\d+) ([-+]?\d*\.\d+) ([-+]?\d*\.\d+) ([-+]?\d*\.\d+) \[(.+)\] (\w+)", line)  # for DQN/Qlearning
            # match = re.match(
            #     r"(\d+) \[(.+)\] (\d+) ([-+]?\d*\.\d+) ([-+]?\d*\.\d+) ([-+]?\d*\.\d+) ([-+]?\d*\.\d+) \[(.+)\] (\w+)",
            #     line)  # for PDQN/MPDQN
            match = re.match(r"(\d+) \[(.+?)\] (\S+) ([-+]?\d*\.\d+) ([-+]?\d*\.\d+) ([-+]?\d*\.\d+) \[(.+?)\] (\w+)",
                             line) # for threshold
            # assert False
            if match != None:
                # Convert the parsing result to the corresponding Python object
                line_data = [int(match.group(1)), json.loads("[" + match.group(2) + "]"), int(match.group(3)),
                             float(match.group(4)), float(match.group(5)), float(match.group(6)),
                             json.loads("[" + match.group(7) + "]"), match.group(8) == "True"]  # for DQN/Qlearning

                # line_data = [int(match.group(1)), json.loads("[" + match.group(2) + "]"), int(match.group(3)),
                #              float(match.group(4)), float(match.group(5)), float(match.group(6)), float(match.group(7)),
                #              json.loads("[" + match.group(8) + "]"), match.group(9) == "True"]  # for PDQN/MPDQN

                parsed_line.append(line_data)

                if match.group(8) == "True":
                    parsed_data.append(parsed_line)
                    parsed_line = []

    return parsed_data
    # return step, replica, cpu_utilization, cpus, reward, resource_use


# Plot --------------------------------------
matplotlib.rcParams.update({'font.size': 15})
# Plot --------------------------------------
def fig_combined_all(x, y2, y1, y3, y4, service_name):
    # Create a figure with two subplots
    fig, (ax4, ax3, ax2) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax1 = ax2.twinx()  # 在同一個 Figure 上建立第一個軸 ax1 (共享 x 軸)

    line_replicas = ax2.plot(x, y1, color="green", linestyle="-.")[0]  # 在第二個軸上繪製綠色的 Replicas 圖形
    line_cpus = ax1.plot(x, y2, color="blue", linestyle="--")[0]  # 在第一個軸上繪製藍色的 Cpus 圖形

    ax2.set_xlabel("step", )  # 設定 x 軸標籤
    ax2.set_ylabel("Replicas", labelpad=30)  # 設定第二個軸的 y 軸標籤
    ax1.set_ylabel("Cpus",)  # 設定第一個軸的 y 軸標籤

    ax2.set_xlim(0, total_episodes * step_per_episodes)  # 設定 x 軸範圍
    ax2.set_ylim(0, 3.5)  # 設定第二個軸的 y 軸範圍
    ax1.set_ylim(0, 1.1)  # 設定第一個軸的 y 軸範圍
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticks([0, 1, 2, 3])


    line_replicas_color = line_replicas.get_color()
    line_replicas_style = line_replicas.get_linestyle()
    line_cpus_color = line_cpus.get_color()
    line_cpus_style = line_cpus.get_linestyle()

    # 建立線條圖示
    replicas_legend = mlines.Line2D([], [], color=line_replicas_color, linestyle=line_replicas_style, label='Replicas')
    cpus_legend = mlines.Line2D([], [], color=line_cpus_color, linestyle=line_cpus_style, label='Cpus')
    # ax2.set_title(service_name, pad=20)
    # Add legends to the first subplot (ax1)
    ax2.legend(handles=[replicas_legend, cpus_legend], loc='upper center', fontsize=12, ncol=2, bbox_to_anchor=(0.5, 1.2))


    # ------------------------------------------------
    # Plot Cpu utilization on the second subplot (ax3)
    ax3.plot(x, y3, color='royalblue')
    ax3.fill_between(x, 0, y3, color='royalblue')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Cpu utilization(%)', labelpad=10)
    ax3.set_ylim(0, 110)
    ax3.set_yticks([0, 25, 50, 75, 100])

    avg = statistics.mean(y3)
    ax3.set_title(service_name + " Avg : " + str(avg), pad=20)
    # -----------------------------------------------
    # Plot additional data on the fourth subplot (ax4)
    ax4.plot(x, y4, color='purple')
    ax4.fill_between(x, 0, y4, color="purple")
    ax4.set_ylabel("Response time(ms)", labelpad=20)
    ax4.set_ylim(0, 50)  # Set y-axis range for the fourth subplot
    ax4.set_yticks([0, 10, 20, 30, 40, 50])
    ax4.set_xlabel('Step')

    avg = statistics.mean(y4)
    ax4.set_title(service_name + " Avg : " + str(avg), pad=20)
    if service_name == "First_level_mn1":
        Rmax = Rmax_mn1
    else:
        Rmax = Rmax_mn2
    ax4.axhline(y=Rmax, color='r', linestyle='--')

    plt.tight_layout()
    plt.savefig(tmp_dir + service_name + "_Combined_all.png", dpi=300)
    plt.show()
def fig_add_Cpus_Replicas(x, y2, y1, service_name):
    fig, ax2 = plt.subplots(figsize=(10, 4))  # 建立一個 Figure 和第二個軸 ax2
    ax1 = ax2.twinx()  # 在同一個 Figure 上建立第一個軸 ax1 (共享 x 軸)

    line_replicas = ax2.plot(x, y1, color="green", linestyle="-.")[0]  # 在第二個軸上繪製綠色的 Replicas 圖形
    line_cpus = ax1.plot(x, y2, color="blue", linestyle="--")[0]  # 在第一個軸上繪製藍色的 Cpus 圖形

    ax2.set_xlabel("step", )  # 設定 x 軸標籤
    ax2.set_ylabel("Replicas", )  # 設定第二個軸的 y 軸標籤
    ax1.set_ylabel("Cpus", )  # 設定第一個軸的 y 軸標籤

    ax2.set_xlim(0, total_episodes * step_per_episodes)  # 設定 x 軸範圍
    ax2.set_ylim(0, 3.5)  # 設定第二個軸的 y 軸範圍
    ax1.set_ylim(0, 1.1)  # 設定第一個軸的 y 軸範圍

    ax2.tick_params(axis="both", labelsize=15)
    ax2.set_yticks([0, 1, 2, 3])
    ax1.tick_params(axis="both", labelsize=15)

    line_replicas_color = line_replicas.get_color()
    line_replicas_style = line_replicas.get_linestyle()
    line_cpus_color = line_cpus.get_color()
    line_cpus_style = line_cpus.get_linestyle()

    # 建立線條圖示
    replicas_legend = mlines.Line2D([], [], color=line_replicas_color, linestyle=line_replicas_style, label='Replicas')
    cpus_legend = mlines.Line2D([], [], color=line_cpus_color, linestyle=line_cpus_style, label='Cpus')

    ax2.set_title(service_name, pad=20)  # 設定圖的標題
    # 建立圖例
    handles = [replicas_legend, cpus_legend]
    labels = [handle.get_label() for handle in handles]
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)
    plt.tight_layout()
    plt.savefig(tmp_dir + service_name + "_Combined.png", dpi=300)
    plt.show()


def fig_add_Cpus(x, y, service_name):
    # plt.figure()
    plt.plot(x, y, color="blue")  # color=color
    plt.title(service_name)
    #　plt.xlabel("step")
    plt.xlabel("step", )
    plt.ylabel("Cpus", )
    # plt.grid(True)
    avg = sum(y) / len(y)
    with open(tmp_dir + 'paper_data.txt', 'a') as file:
        file.write(service_name + " Avg_Cpus: " + str(avg) + "\n")
    plt.xlim(0, total_episodes*step_per_episodes)
    plt.ylim(0, 1.1)
    plt.xticks()
    plt.yticks()
    plt.savefig(tmp_dir + service_name + "_Cpus.png", dpi=300)
    plt.tight_layout()
    plt.show()


def fig_add_Replicas(x, y, service_name):
    # plt.figure()
    plt.plot(x, y, color="green")  # color=color
    plt.title(service_name)
    plt.xlabel("step", )
    plt.ylabel("Replicas", )
    # plt.grid(True)
    avg = sum(y) / len(y)
    with open(tmp_dir + 'paper_data.txt', 'a') as file:
        file.write(service_name + " Avg_Replicas: " + str(avg) + "\n")
    plt.xlim(0, total_episodes*step_per_episodes)
    plt.ylim(0, 4)
    plt.xticks()
    plt.yticks(ticks=[0, 1, 2, 3, 4])
    plt.tight_layout()
    plt.savefig(tmp_dir + service_name + "_Replicas.png", dpi=300)
    plt.show()


def fig_add_Cpu_utilization(x, y, y_, service_name):
    plt.figure(figsize=(10, 4))
    if not if_evaluation:
        plt.plot(x, y, color='royalblue', alpha=0.2)  # color=color # label=label
        plt.plot(x, y_, color='royalblue')  # color=color

    else:
        plt.plot(x, y, color='royalblue')  # color=color # label=label
        plt.fill_between(x, 0, y, color='royalblue')

    avg = sum(y) / len(y)
    plt.title(service_name + " Avg : " + str(avg))
    plt.xlabel("step", )
    plt.ylabel("Cpu utilization(%)", )
    # plt.grid(True)
    plt.xlim(0, total_episodes*step_per_episodes)
    plt.ylim(0, 110)
    plt.xticks()
    plt.yticks(ticks=[0, 25, 50, 75, 100], )
    plt.tight_layout()
    plt.savefig(tmp_dir + service_name + "_Cpu_utilization.png", dpi=300)
    plt.show()


def fig_add_response_times(x, y, y_, service_name):
    plt.figure(figsize=(10, 4))
    if not if_evaluation:
        plt.plot(x, y, color="purple", alpha=0.2)  # color=color # label=label
        plt.plot(x, y_, color="purple")  # color=color # label=label
    else:
        plt.plot(x, y, color="purple")  # color=color # label=label
        plt.fill_between(x, 0, y, color="purple")
    avg = sum(y) / len(y)
    median = statistics.median(y)
    print("median response time", median)
    plt.title(service_name + " Avg : " + str(avg))
    plt.xlabel("step", )
    plt.ylabel("Response time", )

    if service_name == "First_level_mn1":
        Rmax = Rmax_mn1
    else:
        Rmax = Rmax_mn2

    result2 = filter(lambda v: v > Rmax, y)
    R = len(list(result2)) / len(y)
    print("Rmax violation: ", R)
    with open(tmp_dir + 'paper_data.txt', 'a') as file:
        file.write(service_name + " Median: " + str(median) + "\n")
        file.write(service_name + " Tmax_violation: " + str(R) + "\n")
    # plt.grid(True)
    plt.axhline(y=Rmax_mn1, color='r', linestyle='--')
    plt.xlim(0, total_episodes*step_per_episodes)
    plt.ylim(0, 50)
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    plt.savefig(tmp_dir + service_name + "_Response_time.png", dpi=300)

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
    # avg = round(avg, 2)
    print(service_name + " Avg_Resource_use", avg)

    plt.title(service_name + " Avg : " + str(avg))
    plt.xlabel("step", )
    plt.ylabel("Resource_use", )
    # plt.grid(True)
    plt.xlim(0, total_episodes*step_per_episodes)
    plt.ylim(0, 3)
    plt.xticks()
    plt.yticks()
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
    plt.title(service_name + " Avg : " + str(avg), fontsize=12)
    plt.xlabel("step", fontsize=12)
    plt.ylabel("Reward", fontsize=12)

    # plt.grid(True)
    plt.xlim(0, total_episodes*step_per_episodes)
    plt.ylim(-0.6, 0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(tmp_dir + service_name + "_cost.png", dpi=300)
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
            step.append(tmp_step)
            replicas.append(parsed_line[1][0])
            cpu_utilization.append(parsed_line[1][1]*100)
            cpus.append(parsed_line[1][2])
            response_times.append(parsed_line[1][3])
            reward.append(parsed_line[3])  # cost = -reward
            tmp_step += 1
            if tmp_step == step_per_episodes:
                step.append(tmp_step)
                replicas.append(parsed_line[6][0])
                cpu_utilization.append(parsed_line[6][1] * 100)
                cpus.append(parsed_line[6][2])
                response_times.append(parsed_line[6][3])
        # episode_reward.append(sum(reward)/len(reward))
    print(cpu_utilization)
    resource_use = [x * y for x, y in zip(replicas, cpus)]
    replicas_ = moving_average(replicas)
    response_times_ = moving_average(response_times)
    cpu_utilization_ = moving_average(cpu_utilization)
    cpus_ = moving_average(cpus)
    reward_ = moving_average(reward)
    resource_use_ = moving_average(resource_use)
    fig_add_Cpus(step, cpus, service_name)
    fig_add_Replicas(step, replicas, service_name)
    fig_add_Cpus_Replicas(step, cpus, replicas, service_name)
    fig_add_response_times(step, response_times, response_times_, service_name)
    fig_add_Cpu_utilization(step, cpu_utilization, cpu_utilization_, service_name)
    fig_add_Resource_use(step, resource_use, resource_use_, service_name, tmp_dir)
    fig_add_reward(step, reward, reward_, service_name)
    fig_combined_all(step, cpus, replicas, cpu_utilization, response_times, service_name)

tmp_count = 0
for p in path_list:
    # print(p)
    # f = open(p, "r")
    episods_data = parse(p)
    # step, replica, cpu_utilization, cpus, reward, resource_use = parse_episods_data(episods_data)
    # print(len(episods_data))
    parse_episods_data(episods_data, service[tmp_count])

    #step = [x * 30 for x in step]
    # print(y)

    ### plot delay
    # fig_add(step, reward, service[tmp_count])
    # fig_add(x, y2, service[tmp_count])
    # fig_add(x, y3, service[tmp_count])
    tmp_count += 1







