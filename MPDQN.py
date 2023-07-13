import sys

import numpy as np
import random
import requests
import time
import threading
import subprocess
import json
import statistics
import os
import datetime
import math
from pdqn_v1 import PDQNAgent
from pdqn_multipass import MultiPassPDQNAgent
from concurrent.futures import ThreadPoolExecutor, as_completed
print(datetime.datetime.now())

# Need modify ip if ip change
# check cmd : sudo docker-machine ls
ip = "192.168.99.128"  # app_mn1
ip1 = "192.168.99.129"  # app_mn2


# request rate r
data_rate = 50      # if not use_tm
use_tm = 0          # if use_tm
tm_path = 'request/request20.txt'  # traffic path
result_dir = "./mpdqn_result/result18/evaluate2/"

## initial
request_num = []
# timestamp    :  0, 1, 2, , ..., 61, ..., 3601
# learning step:   0,  ..., 1,     , 120

monitor_period = 60
simulation_time = 3600  #
request_n = simulation_time + monitor_period  # for last step
# initial mn1 replica , initial mn2 replica, initial mn1 cpus, initial mn2 cpus
ini_replica1, ini_cpus1, ini_replica2, ini_cpus2 = 1, 1, 1, 1


## manual action for evaluation
## if training : Need modify manual_action to 0
manual_action = 1

## global variable
change = 0   # 1 if take action / 0 if init or after taking action
reset_complete = 0
send_finish = 0
timestamp = 0  # plus 1 in funcntion : send_request
RFID = 0  # oneM2M resource name  (Need different)
event_mn1 = threading.Event()
event_mn2 = threading.Event()
event_timestamp_Ccontrol = threading.Event()


# Parameter
w_pref = 0.5   # 0.8
w_res = 0.5    # 0.2
Tmax_mn1 = 20
Tmax_mn2 = 10
T_upper = 50
error_rate = 0.2  # 0.2
## Learning parameter
# S ={k, u , c, r} {k, u , c}
# k (replica): 1 ~ 3                          actual value : same
# u (cpu utilization) : 0.0, 0.1 0.2 ...1     actual value : 0 ~ 100
# c (used cpus) : 0.1 0.2 ... 1               actual value : same

total_episodes = 1   # Training_episodes

if_test = True
if if_test:
    total_episodes = 1  # Testing_episodes

multipass = True  # False : PDQN  / Ture: MPDQN

# totoal step = episode per step * episode; ex : 60 * 16 = 960
# Exploration parameters
epsilon_steps = 840  #
epsilon_initial = 1   #
epsilon_final = 0.01  # 0.01

# Learning rate
learning_rate_actor_param = 0.0001  # actor # 0.001
learning_rate_actor = 0.001         # critic # 0.01
# Target Learning rate
tau_actor_param = 0.001    # actor  # 0.01
tau_actor = 0.01           # critic # 0.1

gamma = 0.9               # Discounting rate
replay_memory_size = 960  # Replay memory
batch_size = 16
initial_memory_threshold = 16  # Number of transitions required to start learning
use_ornstein_noise = False
layers = [64,]
seed = 7

clip_grad = 0 # no use now
action_input_layer = 0  # no use now
# cres_norml = False

# check result directory
if os.path.exists(result_dir):
    print("Deleting existing result directory...")
    raise SystemExit  # end process

# build dir
os.mkdir(result_dir)

# store setting
path = result_dir + "setting.txt"

# Define settings dictionary
settings = {
    'date': datetime.datetime.now(),
    'data_rate': data_rate,
    'use_tm': use_tm,
    'Tmax_mn1': Tmax_mn1,
    'Tmax_mn2': Tmax_mn2,
    'simulation_time': simulation_time,
    'tau_actor': tau_actor,
    'tau_actor_param': tau_actor_param,
    'learning_rate_actor': learning_rate_actor,
    'learning_rate_actor_param': learning_rate_actor_param,
    'gamma': gamma,
    'epsilon_steps': epsilon_steps,
    'epsilon_final': epsilon_final,
    'replay_memory_size': replay_memory_size,
    'batch_size': batch_size,
    'loss_function': 'MSE loss',
    'layers': layers,
    'if_test': if_test,
    'w_pref': w_pref,
    'w_res': w_res,
}

# Write settings to file
with open(result_dir + 'setting.txt', 'a') as f:
    for key, value in settings.items():
        f.write(f'{key}: {value}\n')


## 8 sensors
sensors = ["RFID_Container_for_stage0", "RFID_Container_for_stage1", "Liquid_Level_Container", "RFID_Container_for_stage2",
         "Color_Container", "RFID_Container_for_stage3", "Contrast_Data_Container", "RFID_Container_for_stage4"]

if use_tm:
    f = open(tm_path)

    for line in f:
        if len(request_num) < request_n:

            request_num.append(int(float(line)))
else:
    request_num = [data_rate for i in range(request_n)]

print("request_num:: ", len(request_num), "simulation_time:: ", simulation_time)


class Env:

    def __init__(self, service_name):

        self.service_name = service_name
        self.cpus = 1
        self.replica = 1
        self.cpu_utilization = 0.0
        self.action_space = ['1', '1', '1']
        self.state_space = [1, 0, 0.5, 20]
        self.n_state = len(self.state_space)
        self.n_actions = len(self.action_space)

        # four service url
        self.url_list = ["http://" + ip + ":666/~/mn-cse/mn-name/AE1/RFID_Container_for_stage4",
                                    "http://" + ip1 + ":777/~/mn-cse/mn-name/AE2/Control_Command_Container",
                                    "http://" + ip + ":1111/test", "http://" + ip1 + ":2222/test"]

    def reset(self):
        # reset replica and cpus values
        self.replica = 1
        self.cpus = 1
        # if self.service_name == 'app_mn2':
        #     self.replica = 1
        #     self.cpus = 1

        self.state_space[0] = self.replica
        self.state_space[2] = self.cpus

        return self.state_space

    def get_response_time(self):

        path1 = result_dir + self.service_name + "_response.txt"
        f1 = open(path1, 'a')
        RFID = random.randint(1000000, 30000000)
        headers = {"X-M2M-Origin": "admin:admin", "Content-Type": "application/json;ty=4"}
        data = {
            "m2m:cin": {
                "con": "true",
                "cnf": "application/json",
                "lbl": "req",
                "rn": str(RFID),
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
        if self.service_name =='app_mn1':
            worker_name = 'worker'
        else:
            worker_name = 'worker1'
        cmd = "sudo docker-machine ssh " + worker_name + " docker stats --no-stream --format \\\"{{ json . }}\\\" "
        returned_text = subprocess.check_output(cmd, shell=True)
        my_data = returned_text.decode('utf8')
        my_data = my_data.split("}")
        cpu_list = []
        for i in range(len(my_data) - 1):
            # print(my_data[i]+"}")
            my_json = json.loads(my_data[i] + "}")
            name = my_json['Name'].split(".")[0]
            cpu = my_json['CPUPerc'].split("%")[0]
            if float(cpu) > 0 and (name == self.service_name):
                cpu_list.append(float(cpu))
        avg_replica_cpu_utilization = sum(cpu_list)/len(cpu_list)
        return avg_replica_cpu_utilization

    def get_cpu_utilization_from_data(self):
        path = result_dir + self.service_name + '_cpu.txt'
        try:
            f = open(path, "r")
            cpu = []
            time = []
            for line in f:
                s = line.split(' ')
                time.append(float(s[0]))
                cpu.append(float(s[1]))
            # Get last five data
            last_avg_cpu = statistics.mean(cpu[-5:])
            f.close()
        except:
            print('cant open')
        return last_avg_cpu

    def discretize_cpu_value(self, value):
        return int(round(value / 10))

    def step(self, action, event, done):
        global timestamp, send_finish, change

        action_replica = action[0]
        action_cpus = action[1][action_replica][0]
        # manual_action
        if self.service_name == 'app_mn1' and manual_action:
            action_replica = 0  # replica  idx
            action_cpus = 1
        if self.service_name == 'app_mn2' and manual_action:
            action_replica = 0  # replica  idx
            action_cpus = 1

        self.replica = action_replica + 1  # 0 1 2 (index)-> 1 2 3 (replica)
        self.cpus = round(action_cpus, 2)
        # print(self.replica, self.cpus)
        change = 1

        # restart
        cmd = "sudo docker-machine ssh default docker service update --replicas 0 " + self.service_name
        returned_text = subprocess.check_output(cmd, shell=True)
        # do agent action
        cmd1 = "sudo docker-machine ssh default docker service scale " + self.service_name + "=" + str(self.replica)
        cmd2 = "sudo docker-machine ssh default docker service update --limit-cpu " + str(self.cpus) + " " + self.service_name
        returned_text = subprocess.check_output(cmd1, shell=True)
        returned_text = subprocess.check_output(cmd2, shell=True)

        time.sleep(30)  # wait service start

        event.set()

        # time.sleep(monitor_period-6)  # wait for monitor ture value
        while True:
            if ((timestamp+6)%monitor_period == 0):
                break

        response_time_list = []
        # self.cpu_utilization = self.get_cpu_utilization()
        self.cpu_utilization = self.get_cpu_utilization_from_data()

        for i in range(5):
            response_time_list.append(self.get_response_time())
            time.sleep(1)
        mean_response_time = statistics.mean(response_time_list)
        mean_response_time = mean_response_time*1000  # 0.05s -> 50ms

        t_max = 0  # for initial
        if self.service_name == "app_mn1":
            t_max = Tmax_mn1
        elif self.service_name == "app_mn2":
            t_max = Tmax_mn2

        Rt = mean_response_time
        # Cost 1
        # B = 10
        # if Rt > t_max:
        #     c_perf = 1
        # else:
        #     tmp_d = B * (Rt - t_max) / t_max
        #     c_perf = math.exp(tmp_d)

        # Cost 2
        # B = 10
        # target = t_max + 2 * math.log(0.9)
        # c_perf = np.where(Rt <= target, np.exp(B * (Rt - t_max) / t_max), 0.9 + ((Rt - target) / (Tupper - target)) * 0.1)

        # Cost 3
        #
        # B = np.log(1+0.5)/((T_upper-t_max)/t_max)
        # c_perf = np.where(Rt <= t_max, 0, np.exp(B * (Rt - t_max) / t_max) - 0.5)

        # Cost 4
        # delay cost
        B = np.log(1+0.5)/((T_upper-t_max)/t_max)
        c_delay = np.where(Rt <= t_max, 0, np.exp(B * (Rt - t_max) / t_max) - 0.5)

        # cpu_utilization cost
        relative_cpu_utilization = self.cpu_utilization/100/self.cpus
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
        c_perf = max(c_delay, c_utilization)

        # resource cost
        c_res = (self.replica*self.cpus)/3   # replica*self.cpus / Kmax
        next_state = []
        # # k, u, c # r

        # u = self.discretize_cpu_value(self.cpu_utilization)
        next_state.append(self.replica)
        next_state.append(relative_cpu_utilization)
        next_state.append(self.cpus)
        next_state.append(Rt)
        # next_state.append(request_num[timestamp])

        # normalize
        # c_perf = 0 + ((c_perf - math.exp(-Tupper/t_max)) / (1 - math.exp(-Tupper/t_max))) * (1 - 0)  # min max normalize
        # c_res = 0 + ((c_res - (1 / 6)) / (1 - (1 / 6))) * (1 - 0)  # min max normalize
        reward_perf = w_pref * c_perf
        reward_res = w_res * c_res
        reward = -(reward_perf + reward_res)
        return next_state, reward, reward_perf, reward_res

def store_cpu(worker_name):
    global timestamp, cpus, change, reset_complete

    cmd = "sudo docker-machine ssh " + worker_name + " docker stats --no-stream --format \\\"{{ json . }}\\\" "
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
                    path = result_dir + name + "_cpu.txt"
                    f = open(path, 'a')
                    data = str(timestamp) + ' '
                    data = data + str(cpu) + ' ' + '\n'

                    f.write(data)
                    f.close()


# reset Environment

def store_reward(service_name, reward):
    # Write the string to a text file
    path = result_dir + service_name + "_reward.txt"
    f = open(path, 'a')
    data = str(reward) + '\n'
    f.write(data)


def store_trajectory(service_name, step, s, a_r, a_c, r, r_perf, r_res, s_, done, if_epsilon):
    path = result_dir + service_name + "_trajectory.txt"
    tmp_s = list(s)
    tmp_s_ = list(s_)
    a_c_ = list(a_c)
    f = open(path, 'a')
    data = str(step) + ' ' + str(tmp_s) + ' ' + str(a_r) + ' ' + str(a_c_) + ' ' + str(r) + ' ' + str(r_perf) + ' ' + str(r_res) + ' ' + str(tmp_s_) + ' ' + str(done) + ' ' + str(if_epsilon) + '\n'
    f.write(data)

def store_error_count(error):
    # Write the string to a text file
    path = result_dir + "error.txt"
    f = open(path, 'a')
    data = str(error) + '\n'
    f.write(data)

def store_request(response_list, response_time_list, timestamp_list):
    # Write the string to a text file
    path = result_dir + "request.txt"
    data = ""
    f = open(path, 'a')
    for i in range(len(response_list)):
        data += str(response_list[i]) + " " + str(response_time_list[i]) + " " + str(timestamp_list[i]) + '\n'
    f.write(data)

def post(url):
    RFID = random.randint(0, 1000000)

    if error_rate > random.random():
        content = "false"
    else:
        content = "true"
    headers = {"X-M2M-Origin": "admin:admin", "Content-Type": "application/json;ty=4"}
    data = {
        "m2m:cin": {
            "con": content,
            "cnf": "application/json",
            "lbl": "req",
            "rn": str(RFID),
        }
    }
    url1 = url + sensors[random.randint(0, 7)]

    s_time = time.time()
    try:
        response = requests.post(url1, headers=headers, json=data, timeout=0.1)
        rt = time.time() - s_time
        response = str(response.status_code)
    except requests.exceptions.Timeout:
        response = "timeout"
        rt = 0.1

    return response, rt

response_list = []
response_time_list = []
timestamp_list = []
def post_url(url, rate, timestamp):
    global response_list, response_time_list, timestamp_list


    with ThreadPoolExecutor(max_workers=rate) as executor:

        results = []
        for i in range(rate):
            results.append(executor.submit(post, url))
            time.sleep(1/rate)  # send requests every 1 / rate s

        for result in as_completed(results):
            response, response_time = result.result()
            timestamp_list.append(timestamp)
            response_list.append(response)
            response_time_list.append(response_time)
            # print(type(response.status_code), response_time)
            # if response != "201":
            #     print(response)


def reset(r1, c1, r2, c2):
    print("reset envronment...")
    cmd_list = [
        "sudo docker-machine ssh default docker service update --replicas 0 app_mn1",
        "sudo docker-machine ssh default docker service update --replicas 0 app_mn2",
        "sudo docker-machine ssh default docker service update --replicas " + str(r1) + " app_mn1",
        "sudo docker-machine ssh default docker service update --limit-cpu " + str(c1) + " app_mn1",
        "sudo docker-machine ssh default docker service update --replicas " + str(r2) + " app_mn2",
        "sudo docker-machine ssh default docker service update --limit-cpu " + str(c2) + " app_mn2"
    ]
    def execute_command(cmd):
        return subprocess.check_output(cmd, shell=True)
    for cmd in cmd_list:
        result = execute_command(cmd)
        print(result)

def send_request(request_num, total_episodes):
    global change, send_finish, reset_complete
    global timestamp, use_tm, RFID
    global response_list, response_time_list, timestamp_list
    error = 0

    for episode in range(total_episodes):
        timestamp = 0
        print("episode: ", episode+1)
        print("reset envronment")
        reset_complete = 0
        reset(ini_replica1, ini_cpus1,  ini_replica2, ini_cpus2)  # reset Environment
        time.sleep(70)
        print("reset envronment complete")
        reset_complete = 1
        send_finish = 0
        for i in request_num:
            print('timestamp: ', timestamp)
            event_mn1.clear()  # set flag to false
            event_mn2.clear()
            if ((timestamp) % monitor_period) == 0 and timestamp!=0 :  # every 60s scaling
                event_timestamp_Ccontrol.set()
                print("wait mn1 mn2 step and service scaling ...")
                event_mn1.wait()  # if flag == false : wait, else if flag == True: continue
                event_mn2.wait()
                change = 0
                print("Start Requesting ...")
            event_timestamp_Ccontrol.clear()
            # exp = np.random.exponential(scale=1 / i, size=i)
            url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/"
            try:
                post_url(url, i, timestamp)
            except:
                print("error")
                error += 1
            timestamp += 1


    send_finish = 1
    store_error_count(error)
    store_request(response_list, response_time_list, timestamp_list)



def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)


def mpdqn(total_episodes, batch_size, gamma, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final,
        clip_grad, layers, multipass, action_input_layer, event, service_name, seed, result_dir):
    global timestamp

    env = Env(service_name)

    agent_class = PDQNAgent
    if multipass:
        agent_class = MultiPassPDQNAgent
    agent = agent_class(
                       env.n_state, env.n_actions,
                       batch_size=batch_size,
                       learning_rate_actor=learning_rate_actor,
                       learning_rate_actor_param=learning_rate_actor_param,
                       epsilon_initial=epsilon_initial,
                       epsilon_steps=epsilon_steps,
                       gamma=gamma,
                       tau_actor=tau_actor,
                       tau_actor_param=tau_actor_param,
                       clip_grad=clip_grad,
                       initial_memory_threshold=initial_memory_threshold,
                       use_ornstein_noise=use_ornstein_noise,
                       replay_memory_size=replay_memory_size,
                       epsilon_final=epsilon_final,
                       actor_kwargs={'hidden_layers': layers,
                                     'action_input_layer': action_input_layer},
                       actor_param_kwargs={'hidden_layers': layers,
                                           'squashing_function': True,
                                           'output_layer_init_std': 0.0001},
                       seed=seed,
                       service_name=service_name,
                       result_dir=result_dir)
    # print(agent)

    # init_state = [1, 1.0, 0.5, 20]  # replica / cpu utiliation / cpus / response time
    step = 1
    for episode in range(1, total_episodes+1):
        if if_test:  # Test
            parts = result_dir.rsplit('/', 2)
            result_dir_ = parts[0] + '/'
            # print(result_dir_)
            agent.load_models(result_dir_ + env.service_name + "_" + str(seed))
            agent.epsilon_final = 0.
            agent.epsilon = 0.
            agent.noise = None

        state = env.reset()  # replica / cpu utiliation / cpus / response time

        done = False

        while True:
            # print(timestamp)
            if timestamp == (monitor_period-6):
                # state[1] = (env.get_cpu_utilization() / 100 / env.cpus)
                state[1] = (env.get_cpu_utilization_from_data() / 100 / env.cpus)
                response_time_list = []
                for i in range(5):
                    response_time_list.append(env.get_response_time())
                    time.sleep(1)
                mean_response_time = statistics.mean(response_time_list)
                mean_response_time = mean_response_time * 1000
                Rt = mean_response_time
                state[3] = Rt
                break
        state = np.array(state, dtype=np.float32)
        print("service name:", env.service_name, "initial state:", state)
        print("service name:", env.service_name, " episode:", episode)
        act, act_param, all_action_parameters, if_epsilon = agent.act(state)

        action = pad_action(act, act_param)

        while True:
            if timestamp == 0:
                done = False
            event_timestamp_Ccontrol.wait()

            if (((timestamp) % monitor_period) == 0) and (not done) and timestamp!=0:
                if timestamp == (simulation_time):
                    done = True
                else:
                    done = False

                next_state, reward, reward_perf, reward_res = env.step(action, event, done)
                # print("service name:", env.service_name, "action: ", action[0] + 1, round(action[1][action[0]][0], 2))

                # Covert np.float32
                next_state = np.array(next_state, dtype=np.float32)
                next_act, next_act_param, next_all_action_parameters, if_epsilon = agent.act(next_state)

                print("service name:", env.service_name, "action: ", act + 1, act_param, all_action_parameters, " step: ", step,
                      " next_state: ",
                      next_state, " reward: ", reward, " done: ", done, "epsilon", agent.epsilon)
                store_trajectory(env.service_name, step, state, act + 1, all_action_parameters, reward, reward_perf,
                                 reward_res,
                                 next_state, done, if_epsilon)
                next_action = pad_action(next_act, next_act_param)
                if not if_test:
                    agent.step(state, (act, all_action_parameters), reward, next_state,
                               (next_act, next_all_action_parameters), done)
                act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters

                action = next_action
                state = next_state
                if not if_test:
                    agent.epsilon_decay()

                step += 1
                # event_timestamp_Ccontrol.clear()
                if done:
                    break
    if not if_test:
        agent.save_models(result_dir + env.service_name + "_" + str(seed))


t1 = threading.Thread(target=send_request, args=(request_num, total_episodes, ))
t2 = threading.Thread(target=store_cpu, args=('worker',))
t3 = threading.Thread(target=store_cpu, args=('worker1',))
t4 = threading.Thread(target=mpdqn, args=(total_episodes, batch_size, gamma, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final,
        clip_grad, layers, multipass, action_input_layer, event_mn1, 'app_mn1', seed, result_dir, ))

t5 = threading.Thread(target=mpdqn, args=(total_episodes, batch_size, gamma, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final,
        clip_grad, layers, multipass, action_input_layer, event_mn2, 'app_mn2', seed, result_dir, ))

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

