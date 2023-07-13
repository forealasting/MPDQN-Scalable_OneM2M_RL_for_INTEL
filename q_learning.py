import requests
import time
import threading
import subprocess
import json
import numpy as np
import random
import statistics
import copy
import os
import datetime
import concurrent.futures
import math
print(datetime.datetime.now())

# request rate r
data_rate = 50      # if not use_tm
use_tm = 0  # if use_tm
# define result path
result_dir = "./all_result/qlearning_result5/"


## initial
request_num = []
# timestamp    : 0, 1, 2, 31, ..., 61, ..., 3601
# learning step:          0,  ..., 1,     , 120

simulation_time = 3602  # 0 ~ 3601:  3602
request_n = simulation_time


## global variable
change = 0   # 1 if take action / 0 if init or after taking action
reset_complete = 0
send_finish = 0
timestamp = 0  # plus 1 in funcntion : send_request

event_mn1 = threading.Event()
event_mn2 = threading.Event()
event_timestamp_Ccontrol = threading.Event()
Rmax_mn1 = 30
Rmax_mn2 = 20

# Need modify ip if ip change
ip = "192.168.99.124"  # app_mn1
ip1 = "192.168.99.125"  # app_mn2
error_rate = 0.2  # 0.2/0.5


## Learning parameter
# S ={k, u , c}
# k (replica): 1 ~ 3                          actual value : same
# u (cpu utilization) : 0.0, 0.1 0.2 ...1     actual value : 0 ~ 100
# c (used cpus) : 0.1 0.2 ... 1               actual value : same
# action_space = ['-r', -1, 0, 1, 'r']
total_episodes = 8            # Total episodes
learning_rate = 0.01          # Learning rate
# max_steps = 121              # Max steps per episode
# Exploration parameters
gamma = 0.9                 # Discounting rate
max_epsilon = 1
min_epsilon = 0
epsilon_decay = 1/840
RFID = 0


# check result directory
if os.path.exists(result_dir):
    print("Deleting existing result directory...")
    raise SystemExit  # end process

# build dir
os.mkdir(result_dir)

# store setting
path = result_dir + "setting.txt"
f = open(path, 'a')
data = 'date: ' + str(datetime.datetime.now()) + '\n'
data += 'data_rate: ' + str(data_rate) + '\n'
data += 'use_tm: ' + str(use_tm) + '\n'
data += 'Rmax_mn1 ' + str(Rmax_mn1) + '\n'
data += 'Rmax_mn2 ' + str(Rmax_mn2) + '\n'
data += 'simulation_time ' + str(simulation_time) + '\n\n'
data += 'learning_rate ' + str(learning_rate) + '\n'
data += 'gamma ' + str(gamma) + '\n'
data += 'max_epsilon ' + str(max_epsilon) + '\n'
data += 'min_epsilon ' + str(min_epsilon) + '\n'
data += 'epsilon_decay ' + str(epsilon_decay) + '\n'

f.write(data)
f.close()

##  stage
stage = ["RFID_Container_for_stage0", "RFID_Container_for_stage1", "Liquid_Level_Container", "RFID_Container_for_stage2",
         "Color_Container", "RFID_Container_for_stage3", "Contrast_Data_Container", "RFID_Container_for_stage4"]

if use_tm:
    #   Modify the workload path if it is different
    f = open('request/request10.txt')

    for line in f:
        if len(request_num) < request_n:

            request_num.append(int(line))
else:
    request_num = [data_rate for i in range(simulation_time)]


print("request_num:: ", len(request_num), "simulation_time:: ", simulation_time)


class Env:

    def __init__(self, service_name="app_mn1"):

        self.service_name = service_name
        self.cpus = 0.5
        self.replica = 1
        self.cpu_utilization = 0.0
        self.action_space = ['-r', '-1', '0', '1', 'r']
        self.n_actions = len(self.action_space)

        # Need modify ip if ip change
        self.url_list = url_list = ["http://" + ip + ":666/~/mn-cse/mn-name/AE1/RFID_Container_for_stage4",
                                    "http://" + ip1 + ":777/~/mn-cse/mn-name/AE2/Control_Command_Container",
                                    "http://" + ip + ":1111/test", "http://" + ip1 + ":2222/test"]

    def reset(self):
        cmd = "sudo docker-machine ssh default docker stack rm app"
        subprocess.check_output(cmd, shell=True)
        cmd1 = "sudo docker-machine ssh default docker stack deploy --compose-file docker-compose.yml app"
        subprocess.check_output(cmd1, shell=True)
        time.sleep(60)

    def get_response_time(self):

        path1 = result_dir + self.service_name + "_response.txt"

        f1 = open(path1, 'a')
        RFID = random.randint(0, 1000000)
        headers = {"X-M2M-Origin": "admin:admin", "Content-Type": "application/json;ty=4"}
        data = {
            "m2m:cin": {
                "con": "true",
                "cnf": "application/json",
                "lbl": "req",
                "rn": str(RFID + 1000),
            }
        }
        # URL
        service_name_list = ["app_mn1", "app_mn2"]
        url = self.url_list[service_name_list.index(self.service_name)]
        try:
            start = time.time()
            response = requests.post(url, headers=headers, json=data, timeout=0.05)
            response = response.status_code
            end = time.time()
            response_time = end - start
        except requests.exceptions.Timeout:
            response = "timeout"
            response_time = 0.05

        data1 = str(timestamp) + ' ' + str(response) + ' ' + str(response_time) + ' ' + str(self.cpus) + ' ' + str(self.replica) + '\n'
        f1.write(data1)
        f1.close()
        if str(response) != '201':
            response_time = 0.05

        return response_time

    def get_cpu_utilization(self):
        path = result_dir + self.service_name + '_cpu.txt'
        try:
            f = open(path, "r")
            cpu = []
            time = []
            for line in f:
                s = line.split(' ')
                time.append(float(s[0]))
                cpu.append(float(s[2]))

            last_avg_cpu = statistics.mean(cpu[-5:])
            f.close()

            return last_avg_cpu
        except:

            print('cant open')

    def discretize_cpu_value(self, value):
        return int(round(value / 10))

    def step(self, action_index, event, done):
        global timestamp, send_finish, change, simulation_time

        action = self.action_space[action_index]
        if action == '-r':
            if self.replica > 1:
                self.replica -= 1
                change = 1
                cmd = "sudo docker-machine ssh default docker service scale " + self.service_name + "=" + str(self.replica)
                returned_text = subprocess.check_output(cmd, shell=True)

        if action == '-1':
            if self.cpus >= 0.5:
                self.cpus -= 0.1
                self.cpus = round(self.cpus, 1)  # ex error:  0.7999999999999999
                change = 1
                cmd = "sudo docker-machine ssh default docker service update --limit-cpu " + str(self.cpus) + " " + self.service_name
                returned_text = subprocess.check_output(cmd, shell=True)

        if action == '1':
            if self.cpus < 1:
                self.cpus += 0.1
                self.cpus = round(self.cpus, 1)
                change = 1
                cmd = "sudo docker-machine ssh default docker service update --limit-cpu " + str(self.cpus) + " " + self.service_name
                returned_text = subprocess.check_output(cmd, shell=True)

        if action == 'r':
            if self.replica < 3:
                self.replica += 1
                change = 1
                cmd = "sudo docker-machine ssh default docker service scale " + self.service_name + "=" + str(self.replica)
                returned_text = subprocess.check_output(cmd, shell=True)

        if self.service_name == 'app_mn1':
            time.sleep(10) # wait app_mn2 service start
        time.sleep(30)  # wait service start

        if not done:
            # print(self.service_name, "_done: ", done)
            # print(self.service_name, "_step complete")
            event.set()

        response_time_list = []
        time.sleep(25)
        for i in range(5):
            time.sleep(1)
            response_time_list.append(self.get_response_time())

        if done:
            # print(self.service_name, "_done: ", done)
            time.sleep(10)
            event.set()  # if done and after get_response_time
        # avg_response_time = sum(response_time_list)/len(response_time_list)
        # print(response_time_list)
        mean_response_time = statistics.mean(response_time_list)
        mean_response_time = mean_response_time*1000  # 0.05s -> 50ms
        t_max = 0
        if mean_response_time >= 50:
            Rt = 50
        else:
            Rt = mean_response_time
        if self.service_name == "app_mn1":
            t_max = Rmax_mn1
        elif self.service_name == "app_mn2":
            t_max = Rmax_mn2


        tmp_d = math.exp(50 / t_max)
        tmp_n = math.exp(Rt / t_max)
        c_perf = tmp_n / tmp_d

        c_res = (self.replica*self.cpus)/3   # replica*self.cpus / Kmax
        next_state = []
        # k, u, c # r
        self.cpu_utilization = self.get_cpu_utilization()
        path = result_dir + self.service_name + "_agent_get_cpu.txt"
        f1 = open(path, 'a')
        data = str(timestamp) + ' ' + str(self.cpu_utilization) + '\n'
        f1.write(data)
        f1.close()

        u = self.discretize_cpu_value(self.cpu_utilization)
        next_state.append(self.replica)
        next_state.append(u/10/self.cpus)
        next_state.append(self.cpus)
        # state.append(req)

        # cost function
        w_pref = 0.5
        w_res = 0.5
        c_perf = 0 + ((c_perf - math.exp(-50 / t_max)) / (1 - math.exp(-50 / t_max))) * (1 - 0)
        c_res = 0 + ((c_res - (1 / 6)) / (1 - (1 / 6))) * (1 - 0)
        reward_perf = w_pref * c_perf
        reward_res = w_res * c_res
        reward = -(reward_perf + reward_res)
        return next_state, reward, reward_perf, reward_res


class QLearningTable:

    def __init__(self, actions, learning_rate=0.01, gamma=0.9, max_epsilon=1, min_epsilon=0.1, epsilon_decay=1 / 300):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.full((10, 11, 10, 5), -np.iinfo(np.int32).max)  # -2147483647

    def choose_action(self, state):
        available_actions = self.get_available_actions(state)
        s = copy.deepcopy(state)
        s[2] = int(s[2] * 10 - 1)
        s[1] = int(s[1])
        s[0] = int(s[0])-1
        # action selection
        if self.epsilon > np.random.uniform():
            # choose random action
            action = np.random.choice(available_actions)
        else:
            # choose greedy action
            q_values = self.q_table[s[0], s[1], s[2], :]
            q_values[np.isin(range(5), available_actions, invert=True)] = -np.iinfo(np.int32).max
            action = np.argmax(q_values)

        return action

    def learn(self, state, a, r, next_state, done):
        s = copy.deepcopy(state)
        s_ = copy.deepcopy(next_state)
        # state  = [1, 0.0, 0.5]
        # transform state to index
        s[2] = int(s[2] * 10 - 1)
        s[1] = int(s[1])
        s[0] = int(s[0]) - 1

        s_[2] = int(s_[2] * 10 - 1)
        s_[1] = int(s_[1])
        s_[0] = int(s_[0])-1

        q_predict = self.q_table[s[0], s[1], s[2], a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * np.max(self.q_table[s_[0], s_[1], s_[2], :])
        self.q_table[s[0], s[1], s[2], a] = q_predict + self.lr * (q_target - q_predict)

        # linearly decrease epsilon
        self.epsilon = max(
            self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
            ) * self.epsilon_decay
        )

    def get_available_actions(self, state):
        # S ={k, u , c}
        # k (replica): 1 ~ 3                          actual value : same
        # u (cpu utilization) : 0.0, 0.1 0.2 ...1     actual value : 0 ~ 100
        # c (used cpus) : 0.1 0.2 ... 1               actual value : same
        # action_space = ['-r', -1, 0, 1, 'r']        r : replica   1: cpus

        actions = [0, 1, 2, 3, 4]  # action index
        if state[0] == 1:
            actions.remove(0)
        if state[0] == 3:
            actions.remove(4)
        if state[2] == 0.5:
            actions.remove(1)
        if state[2] == 1:
            actions.remove(3)

        return actions


def store_cpu(start_time, woker_name):
    global timestamp, cpus, change, reset_complete

    cmd = "sudo docker-machine ssh " + woker_name + " docker stats --all --no-stream --format \\\"{{ json . }}\\\" "
    while True:

        if send_finish == 1:
            break
        if change == 0 and reset_complete == 1:
            returned_text = subprocess.check_output(cmd, shell=True)
            my_data = returned_text.decode('utf8')
            # print(my_data.find("CPUPerc"))
            my_data = my_data.split("}")
            # state_u = []
            for i in range(len(my_data) - 1):
                # print(my_data[i]+"}")
                my_json = json.loads(my_data[i] + "}")
                name = my_json['Name'].split(".")[0]
                cpu = my_json['CPUPerc'].split("%")[0]
                if float(cpu) > 0:
                    final_time = time.time()
                    t = final_time - start_time
                    path = result_dir + name + "_cpu.txt"
                    f = open(path, 'a')
                    data = str(timestamp) + ' ' + str(t) + ' '
                    # for d in state_u:
                    data = data + str(cpu) + ' ' + '\n'

                    f.write(data)
                    f.close()


# reset Environment
def reset():
    cmd1 = "sudo docker-machine ssh default docker service scale app_mn1=1"
    cmd2 = "sudo docker-machine ssh default docker service scale app_mn2=1"
    cmd3 = "sudo docker-machine ssh default docker service update --limit-cpu 0.5 app_mn1"
    cmd4 = "sudo docker-machine ssh default docker service update --limit-cpu 0.5 app_mn2"
    subprocess.check_output(cmd1, shell=True)
    subprocess.check_output(cmd2, shell=True)
    subprocess.check_output(cmd3, shell=True)
    subprocess.check_output(cmd4, shell=True)


def store_reward(service_name, reward):
    # Write the string to a text file
    path = result_dir + service_name + "_reward.txt"
    f = open(path, 'a')
    data = str(reward) + '\n'
    f.write(data)

def store_trajectory(service_name, step, s, a, r, r_perf, r_res, s_, done):
    path = result_dir + service_name + "_trajectory.txt"
    tmp_s = list(s)
    tmp_s_ = list(s_)
    f = open(path, 'a')
    data = str(step) + ' ' + str(tmp_s) + ' ' + str(a) + ' ' + str(r) + ' ' + str(r_perf) + ' ' + str(r_res) + ' ' + str(tmp_s_) + ' ' + str(done) + '\n'
    f.write(data)


def store_error_count(error):
    # Write the string to a text file
    path = result_dir + "error.txt"
    f = open(path, 'a')
    data = str(error) + '\n'
    f.write(data)


def post_url(url, RFID, content):

    headers = {"X-M2M-Origin": "admin:admin", "Content-Type": "application/json;ty=4"}
    data = {
        "m2m:cin": {
            "con": content,
            "cnf": "application/json",
            "lbl": "req",
            "rn": str(RFID),
        }
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=0.05)
        response = str(response.status_code)
    except requests.exceptions.Timeout:
        response = "timeout"

    return response



def send_request(stage, request_num, start_time, total_episodes):
    global change, send_finish, reset_complete
    global timestamp, use_tm, RFID
    error = 0
    for episode in range(total_episodes):
        print("episode: ", episode)
        print("reset envronment")
        reset_complete = 0
        reset()  # reset Environment
        time.sleep(70)
        print("reset envronment complete")
        reset_complete = 1
        send_finish = 0
        timestamp = 0
        for i in request_num:
            # print("timestamp: ", timestamp)
            event_mn1.clear()
            event_mn2.clear()
            if ((timestamp - 1) % 30) == 0:
                print("wait mn1 mn2 step ...")
                event_mn1.wait()
                event_mn2.wait()
                change = 0
            event_timestamp_Ccontrol.clear()
            exp = np.random.exponential(scale=1 / i, size=i)
            tmp_count = 0
            for j in range(i):
                try:
                    url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/"
                    # change stage
                    url1 = url + stage[(tmp_count * 10 + j) % 8]
                    if error_rate > random.random():
                        content = "false"
                    else:
                        content = "true"
                    s_time = time.time()
                    response = post_url(url1, RFID, content)
                    t_time = time.time()
                    rt = t_time - s_time
                    RFID += 1

                except:
                    print("eror")
                    error += 1

                if use_tm == 1:
                    time.sleep(exp[tmp_count])
                    tmp_count += 1

                else:
                    time.sleep(1 / i)  # send requests every 1s

            timestamp += 1
            event_timestamp_Ccontrol.set()

    send_finish = 1
    final_time = time.time()
    alltime = final_time - start_time
    store_error_count(error)
    print('time:: ', alltime)


def q_learning(total_episodes, learning_rate, gamma, max_epsilon, min_epsilon, epsilon_decay, event, service_name):
    global timestamp, simulation_time, change, send_finish

    env = Env(service_name)
    actions = list(range(env.n_actions))
    RL = QLearningTable(actions, learning_rate, gamma, max_epsilon, min_epsilon, epsilon_decay)
    all_rewards = []
    step = 0
    init_state = [1, 0.0, 0.5]
    done = False

    for episode in range(total_episodes):
        # initial observation
        state = init_state
        rewards = []  # record reward every episode
        done = False
        while True:
            if timestamp == 0:
                done = False
            event_timestamp_Ccontrol.wait()
            if (((timestamp - 1) % 30) == 0) and (not done):
                # RL choose action based on state
                action = RL.choose_action(state)

                # agent take action and get next state and reward
                # print("service_name: ", service_name, " timestamp: ", timestamp, " step: ", step)
                if timestamp == (simulation_time-1):
                    done = True
                else:
                    done = False

                next_state, reward, reward_perf, reward_res = env.step(action, event, done)
                print(service_name, "action: ", action, " step: ", step, " next_state: ", next_state, " reward: ", reward, " done: ", done)

                store_trajectory(env.service_name, step, state, action, reward, reward_perf, reward_res, next_state, done)
                rewards.append(reward)
                # RL learn from this transition
                RL.learn(state, action, reward, next_state, done)

                # swap state
                state = next_state
                step += 1
                if done:
                    avg_rewards = sum(rewards)/len(rewards)
                    break
                event_timestamp_Ccontrol.clear()

        store_reward(service_name, avg_rewards)
        all_rewards.append(avg_rewards)
    # episode end
    print("service:", service_name, all_rewards)


start_time = time.time()

t1 = threading.Thread(target=send_request, args=(stage, request_num, start_time, total_episodes, ))
t2 = threading.Thread(target=store_cpu, args=(start_time, 'worker',))
t3 = threading.Thread(target=store_cpu, args=(start_time, 'worker1',))
t4 = threading.Thread(target=q_learning, args=(total_episodes, learning_rate, gamma, max_epsilon, min_epsilon, epsilon_decay, event_mn1, 'app_mn1', ))
t5 = threading.Thread(target=q_learning, args=(total_episodes, learning_rate, gamma, max_epsilon, min_epsilon, epsilon_decay, event_mn2, 'app_mn2', ))


t1.start()
t2.start()
t3.start()
t4.start()
t5.start()


t1.join()
t2.join()
t3.join()
t4.join()
t5.join()

