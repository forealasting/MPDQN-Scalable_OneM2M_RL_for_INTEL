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
import re

print(datetime.datetime.now())

# request rate r
data_rate = 50      # if not use_tm
use_tm = 1  # if use_tm
result_dir = "./mpdqn_result/result5/"

tmp_dir = "mpdqn_database/database4"
# tmp_dir = "mpdqn_result/result1"
path1 = tmp_dir + "/app_mn1_trajectory.txt"
path2 = tmp_dir + "/app_mn2_trajectory.txt"
## initial
request_num = []
# timestamp    : 0, 1, 2, 31, ..., 61, ..., 3601
# learning step:          0,  ..., 1,     , 120

simulation_time = 3602  # 300 s  # 0 ~ 3601:  3602
request_n = simulation_time

## global variable
change = 0   # 1 if take action / 0 if init or after taking action
reset_complete = 0
send_finish = 0
timestamp = 0  # plus 1 in funcntion : send_request
RFID = 0  # random number for data
event_mn1 = threading.Event()
event_mn2 = threading.Event()
event_timestamp_Ccontrol = threading.Event()

# Need modify ip if ip change
ip = "192.168.99.124"  # app_mn1
ip1 = "192.168.99.125"  # app_mn2
error_rate = 0.2  # 0.2/0.5
Rmax_mn1 = 20
Rmax_mn2 = 20
learning_step = 2000  # 960
offline_learning_complete = True
## Learning parameter
# S ={k, u , c, r}
# k (replica): 1 ~ 3                          actual value : same
# u (cpu utilization) : 0.0, 0.1 0.2 ...1     actual value : 0 ~ 100
# c (used cpus) : 0.1 0.2 ... 1               actual value : same
# action_space = ['-r', -1, 0, 1, 'r']
Training_episodes = 1
Test_episodes = 0
if_test = False
total_episodes = Training_episodes + Test_episodes      # Total episodes
multipass = True  # False : PDQN  / Ture: MPDQN
# Exploration parameters
gamma = 0.9                 # Discounting rate

# Exploration parameters
epsilon_steps = 840
epsilon_initial = 1
epsilon_final = 0.01

# Learning rate
tau_actor = 0.1
tau_actor_param = 0.001
learning_rate_actor = 0.001
learning_rate_actor_param = 0.0001

replay_memory_size = 960  # Replay memory
batch_size = 16
initial_memory_threshold = 8  # Number of transitions required to start learning
use_ornstein_noise = False
clip_grad = 10
layers = [64,]
seed = 16

action_input_layer = 0  # no use
# check result directory
# if os.path.exists(result_dir):
#     print("Existing result directory...")
#     raise SystemExit  # end process
#
# # build dir
# os.mkdir(result_dir)
# store setting
path = result_dir + "setting.txt"

# Define settings dictionary
settings = {
    'date': datetime.datetime.now(),
    'data_rate': data_rate,
    'use_tm': use_tm,
    'Rmax_mn1': Rmax_mn1,
    'Rmax_mn2': Rmax_mn2,
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
    'layers': layers
}

# Write settings to file
with open(result_dir + 'setting.txt', 'a') as f:
    for key, value in settings.items():
        f.write(f'{key}: {value}\n')


## 7/8 stage
stage = ["RFID_Container_for_stage0", "RFID_Container_for_stage1", "Liquid_Level_Container", "RFID_Container_for_stage2",
         "Color_Container", "RFID_Container_for_stage3", "Contrast_Data_Container", "RFID_Container_for_stage4"]

if use_tm:
    f = open('request/request12.txt')

    for line in f:
        if len(request_num) < request_n:

            request_num.append(int(float(line)*2))
else:
    request_num = [data_rate for i in range(simulation_time)]

print("request_num:: ", len(request_num), "simulation_time:: ", simulation_time)


def parse(p):
    with open(p, "r") as f:
        data = f.read().splitlines()
        parsed_data = []
        parsed_line = []

        for line in data:
            # parse data
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

                line_data = [int(match.group(1)), json.loads("[" + match.group(2) + "]"), int(match.group(3)),
                             json.loads("[" + match.group(4) + "]"), float(match.group(5)), float(match.group(6)),
                             float(match.group(7)), json.loads("[" + match.group(8) + "]"), match.group(9) == "True"]
                parsed_line.append(line_data)

                parsed_line.append(line_data)
                # 9 8
                if match.group(9) == "True":
                    parsed_data.append(parsed_line)
                    parsed_line = []

    return parsed_data

class Env:

    def __init__(self, service_name="app_mn1"):

        self.service_name = service_name
        self.cpus = 0.5
        self.replica = 1
        self.cpu_utilization = 0.0
        self.action_space = ['1', '1', '1']
        self.state_space = [1, 0.0, 0.5]  # [1, 0.0, 0.5, 10]
        self.n_state = len(self.state_space)
        self.n_actions = len(self.action_space)

        # Need modify ip if container name change
        self.url_list = ["http://" + ip + ":666/~/mn-cse/mn-name/AE1/RFID_Container_for_stage4",
                                    "http://" + ip1 + ":777/~/mn-cse/mn-name/AE2/Control_Command_Container",
                                    "http://" + ip + ":1111/test", "http://" + ip1 + ":2222/test"]

    def reset(self):
        self.cpus = 0.5
        self.replica = 1

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
                cpu.append(float(s[1]))

            last_avg_cpu = statistics.mean(cpu[-10:])
            f.close()

            return last_avg_cpu
        except:

            print('cant open')

    def discretize_cpu_value(self, value):
        return int(round(value / 10))

    def step(self, action, event, done):
        global timestamp, send_finish, change, simulation_time


        action_replica = action[0]
        action_cpus = action[1][action_replica][0]
        self.replica = action_replica + 1  # 0 1 2 (index)-> 1 2 3 (replica)
        self.cpus = round(action_cpus, 2)
        # print(self.replica, self.cpus)
        change = 1
        cmd = "sudo docker-machine ssh default docker service scale " + self.service_name + "=" + str(self.replica)
        cmd1 = "sudo docker-machine ssh default docker service update --limit-cpu " + str(self.cpus) + " " + self.service_name
        returned_text = subprocess.check_output(cmd, shell=True)
        returned_text = subprocess.check_output(cmd1, shell=True)

        if self.service_name == 'app_mn1':
            time.sleep(5)  # wait app_mn1 service start
        time.sleep(30)  # wait service start

        if not done:
            # print(self.service_name, "_done: ", done)
            # print(self.service_name, "_step complete")
            event.set()

        response_time_list = []
        time.sleep(20)
        for i in range(5):
            time.sleep(1)
            response_time_list.append(self.get_response_time())

        if done:
            # print(self.service_name, "_done: ", done)
            time.sleep(5)
            event.set()  # if done and after get_response_time
        # mean_response_time = sum(response_time_list)/len(response_time_list)
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
        # # k, u, c # r
        self.cpu_utilization = self.get_cpu_utilization()
        path = result_dir + self.service_name + "_agent_get_cpu.txt"
        f1 = open(path, 'a')
        data = str(timestamp) + ' ' + str(self.cpu_utilization) + '\n'
        f1.write(data)
        f1.close()
        u = self.discretize_cpu_value(self.cpu_utilization)
        next_state.append(self.replica)
        next_state.append(u/10/self.cpus)
        # next_state.append(u/10)
        next_state.append(self.cpus)
        # next_state.append(request_num[timestamp])

        # cost function
        w_pref = 0.5
        w_res = 0.5
        c_perf = 0 + ((c_perf - math.exp(-50/t_max)) / (1 - math.exp(-50/t_max))) * (1 - 0)
        c_res = 0 + ((c_res - (1 / 6)) / (1 - (1 / 6))) * (1 - 0)
        reward_perf = w_pref * c_perf
        reward_res = w_res * c_res
        reward = -(reward_perf + reward_res)
        return next_state, reward, reward_perf, reward_res



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
                    path = result_dir + name + "_cpu.txt"
                    f = open(path, 'a')
                    data = str(timestamp) + ' '
                    # for d in state_u:
                    data = data + str(cpu) + ' ' + '\n'

                    f.write(data)
                    f.close()


# reset Environment
def reset():
    cmd1 = "sudo docker-machine ssh default docker service update --replicas 1 app_mn1 "
    cmd2 = "sudo docker-machine ssh default docker service update --replicas 1 app_mn2 "
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


def store_loss(service_name, loss):
    # Write the string to a text file
    path = result_dir + service_name + "_loss.txt"
    f = open(path, 'a')
    data = str(loss) + '\n'
    f.write(data)


def store_trajectory(service_name, step, s, a_r, a_c, r, r_perf, r_res, s_, done):
    path = result_dir + service_name + "_trajectory.txt"
    tmp_s = list(s)
    tmp_s_ = list(s_)
    a_c_ = list(a_c)
    f = open(path, 'a')
    data = str(step) + ' ' + str(tmp_s) + ' ' + str(a_r) + ' ' + str(a_c_) + ' ' + str(r) + ' ' + str(r_perf) + ' ' + str(r_res) + ' ' + str(tmp_s_) + ' ' + str(done) + '\n'
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
        timestamp = 0
        print("episode: ", episode)
        print("reset envronment")
        reset_complete = 0
        reset()  # reset Environment
        time.sleep(70)
        print("reset envronment complete")
        reset_complete = 1
        send_finish = 0
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
                    response = post_url(url1, RFID, content)
                    RFID += 1

                except:
                    print("error")
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


def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)


def mpdqn(total_episodes, batch_size, gamma, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final,
        clip_grad, layers, multipass, action_input_layer, event, service_name, seed):
    global timestamp, simulation_time

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
                       seed=seed)
    # print(agent)
    # --------------------- offline training
    if not offline_learning_complete:
        tmp_step = 0
        steps = []
        states = []
        acts = []
        all_action_parameterss = []
        rewards = []
        next_states = []
        terminals = []
        if env.service_name == "app_mn1":
            path = path1
        else:
            path = path2
        episods_data = parse(path)
        # print(len(episods_data))
        for episode in range(1, total_episodes + 1):
            for parsed_line in episods_data[episode - 1]:
                # parsed_line = episods_data[episode-1]
                # step.append(parsed_line[0])
                steps.append(tmp_step)
                tmp_step += 1
                states.append(parsed_line[1])
                acts.append(parsed_line[2])
                all_action_parameterss.append(parsed_line[3])
                rewards.append(parsed_line[4])  # cost = -reward
                next_states.append(parsed_line[7])
                terminals.append(parsed_line[8])

        for i in range(len(steps)):
            state = states[i]
            state = np.array(state, dtype=np.float32)
            act = acts[i] - 1
            # print(all_action_parameterss[i])
            all_action_parameters = all_action_parameterss[i]
            # all_action_parameters = np.random.uniform(0.5, 1.0, size=(1, 3))
            # all_action_parameters[0][act - 1] = tmp_parameters
            # all_action_parameters = all_action_parameters.tolist()[0]
            # print(act, all_action_parameters)

            reward = float(rewards[i])

            next_state = next_states[i]
            next_state = np.array(next_state, dtype=np.float32)
            terminal = terminals[i]
            agent._add_sample(state, np.concatenate(([act], all_action_parameters)).ravel(), reward, next_state,
                              terminal=terminal)

        step = 0
        # for episode in range(total_episodes):
        for i in range(learning_step):
            print("service name:", env.service_name, " step:", i)
            agent._optimize_td_loss()
            step += 1
        agent.save_models(result_dir + env.service_name + "_" + str(seed))

    agent.load_models(result_dir + env.service_name + "_" + str(seed))

    start_time = time.time()
    init_state = [1, 0.3, 0.5]  # [1, 0.0, 0.5, 50]
    step = 0
    for episode in range(1, total_episodes+1):
        if (episode == total_episodes) and if_test:  # Test
            agent.epsilon_final = 0.
            agent.epsilon = 0.
            agent.noise = None
        env.reset()
        state = init_state
        done = False
        state = np.array(state, dtype=np.float32)

        print("service name:", env.service_name, " episode:", episode)
        act, act_param, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param)
        # print(action[0], action[1][action[0]][0])

        while True:
            if timestamp == 0:
                done = False
            event_timestamp_Ccontrol.wait()
            if (((timestamp - 1) % 30) == 0) and (not done):
                if timestamp == (simulation_time - 1):
                    done = True
                else:
                    done = False

                next_state, reward, reward_perf, reward_res = env.step(action, event, done)
                # print("service name:", env.service_name, "action: ", action[0] + 1, round(action[1][action[0]][0], 2))

                # Covert np.float32
                next_state = np.array(next_state, dtype=np.float32)
                next_act, next_act_param, next_all_action_parameters = agent.act(next_state)  # next_act: 2 # next_act_param: 0.85845 # next_all_action_parameters: -0.79984,-0.97112,0.85845
                print("service name:", env.service_name, "action: ", act + 1, act_param, all_action_parameters, " step: ", step,
                      " next_state: ",
                      next_state, " reward: ", reward, " done: ", done, "epsilon", agent.epsilon)
                store_trajectory(env.service_name, step, state, act + 1, all_action_parameters, reward, reward_perf,
                                 reward_res,
                                 next_state, done)
                next_action = pad_action(next_act, next_act_param)
                # agent.step(state, (act, all_action_parameters), reward, next_state,
                #            (next_act, next_all_action_parameters), done)
                act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
                #
                action = next_action
                state = next_state
                # agent.epsilon_decay()

                step += 1
                event_timestamp_Ccontrol.clear()
            if done:
                break

    # agent.save_models(result_dir)
    end_time = time.time()
    print(end_time-start_time)


start_time = time.time()

t1 = threading.Thread(target=send_request, args=(stage, request_num, start_time, total_episodes, ))
t2 = threading.Thread(target=store_cpu, args=(start_time, 'worker',))
t3 = threading.Thread(target=store_cpu, args=(start_time, 'worker1',))
t4 = threading.Thread(target=mpdqn, args=(total_episodes, batch_size, gamma, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final,
        clip_grad, layers, multipass, action_input_layer, event_mn1, 'app_mn1', seed, ))

t5 = threading.Thread(target=mpdqn, args=(total_episodes, batch_size, gamma, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final,
        clip_grad, layers, multipass, action_input_layer, event_mn2, 'app_mn2', seed, ))

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

