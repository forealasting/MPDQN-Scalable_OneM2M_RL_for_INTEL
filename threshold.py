import requests
import time
import threading
import subprocess
import json
import numpy as np
import random
import os
import math
import statistics
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

print(datetime.datetime.now())


# Need modify ip if ip change
ip = "192.168.99.128"  # app_mn1
ip1 = "192.168.99.129"  # app_mn2


# request rate r
data_rate = 50      # if not use_tm
use_tm = 1  # if use_tm

# result path
result_dir = "./threshold_result/result2/"
tm_path = 'request/request20.txt'  # traffic path

## initial
request_num = []
# timestamp    :  0, 1, 2, , ..., 61, ..., 3601
# learning step:   0,  ..., 1,     , 120

monitor_period = 60
simulation_time = 3600  #
request_n = simulation_time + monitor_period  # for last step
ini_replica1, ini_cpus1, ini_replica2, ini_cpus2 = 1, 1, 1, 1

## global variable
change = 0   # 1 if take action / 0 if init or after taking action
reset_complete = 0
send_finish = 0
timestamp = 0  # plus 1 in funcntion : send_request
RFID = 0  # random number for data
event_mn1 = threading.Event()
event_mn2 = threading.Event()
event_timestamp_control = threading.Event()


# Parameter
w_pref = 0.8
w_res = 0.2
error_rate = 0.2  # 0.2/0.5
Tmax_mn1 = 20
Tmax_mn2 = 20
T_upper = 50

## Learning parameter
# S ={k, u , c, r} {k, u , c}
# k (replica): 1 ~ 3                          actual value : same
# u (cpu utilization) : 0.0, 0.1 0.2 ...1     actual value : 0 ~ 100
# c (used cpus) : 0.1 0.2 ... 1               actual value : same
# action_space = ['-r', -1, 0, 1, 'r']
total_episodes = 1

seed = 7
np.random.seed(seed)

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
    'w_pref': w_pref,
    'w_res': w_res,
}


# Write settings to file
with open(result_dir + 'setting.txt', 'a') as f:
    for key, value in settings.items():
        f.write(f'{key}: {value}\n')


## 8 stage
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
        self.state_space = [1, 1.0, 1, 20]
        self.n_state = len(self.state_space)
        self.n_actions = len(self.action_space)

        # Need modify ip if container name change
        self.url_list = ["http://" + ip + ":666/~/mn-cse/mn-name/AE1/RFID_Container_for_stage4",
                                    "http://" + ip1 + ":777/~/mn-cse/mn-name/AE2/Control_Command_Container",
                                    "http://" + ip + ":1111/test", "http://" + ip1 + ":2222/test"]

    def reset(self):
        self.replica = 1
        self.cpus = 1
        self.state_space[0] = self.replica
        self.state_space[2] = self.cpus

        return self.state_space
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
        if self.service_name =='app_mn1':
            worker_name = 'worker'
        else:
            worker_name = 'worker1'
        cmd = "sudo docker-machine ssh " + worker_name + " docker stats --all --no-stream --format \\\"{{ json . }}\\\" "
        returned_text = subprocess.check_output(cmd, shell=True)
        my_data = returned_text.decode('utf8')
        my_data = my_data.split("}")
        cpu_list = []
        for i in range(len(my_data) - 1):
            # print(my_data[i]+"}")
            my_json = json.loads(my_data[i] + "}")
            name = my_json['Name'].split(".")[0]
            cpu = my_json['CPUPerc'].split("%")[0]
            if float(cpu) > 0 and name == self.service_name:
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

            last_avg_cpu = statistics.mean(cpu[-5:])
            f.close()
        except:
            print('cant open')
        return last_avg_cpu

    def discretize_cpu_value(self, value):
        return int(round(value / 10))

    def step(self, action, event, done):
        global timestamp, send_finish, change, simulation_time

        if action == '-1':
            if self.replica > 1:
                self.replica -= 1
                change = 1
                cmd = "sudo docker-machine ssh default docker service scale " + self.service_name + "=" + str(
                    self.replica)
                returned_text = subprocess.check_output(cmd, shell=True)

        if action == '+1':
            if self.replica < 3:
                self.replica += 1
                change = 1
                cmd = "sudo docker-machine ssh default docker service scale " + self.service_name + "=" + str(self.replica)
                returned_text = subprocess.check_output(cmd, shell=True)

        else:
            change = 1
            cmd = "sudo docker-machine ssh default docker service update --replicas 0 " + self.service_name
            cmd1 = "sudo docker-machine ssh default docker service update --replicas " + str(self.replica) + " " + self.service_name
            returned_text = subprocess.check_output(cmd, shell=True)
            returned_text = subprocess.check_output(cmd1, shell=True)

        time.sleep(30)  # wait service start

        event.set()

        time.sleep(monitor_period-5)  # wait for monitor ture value

        response_time_list = []

        for i in range(5):
            time.sleep(1)
            response_time_list.append(self.get_response_time())
        mean_response_time = statistics.mean(response_time_list)
        mean_response_time = mean_response_time*1000  # 0.05s -> 50ms

        # self.cpu_utilization = self.get_cpu_utilization()
        self.cpu_utilization = self.get_cpu_utilization_from_data()

        t_max = 0
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
        B = np.log(1+0.5)/((T_upper-t_max)/t_max)
        c_perf = np.where(Rt <= t_max, 0, np.exp(B * (Rt - t_max) / t_max) - 0.5)

        c_res = (self.replica*self.cpus)/3   # replica*self.cpus / Kmax
        next_state = []
        # # k, u, c # r

        # u = self.discretize_cpu_value(self.cpu_utilization)
        next_state.append(self.replica)
        next_state.append(self.cpu_utilization/100/self.cpus)
        next_state.append(self.cpus)
        next_state.append(Rt)
        # next_state.append(request_num[timestamp])

        # cost function

        # c_perf = 0 + ((c_perf - math.exp(-Tupper/t_max)) / (1 - math.exp(-Tupper/t_max))) * (1 - 0)  # min max normalize
        # c_res = 0 + ((c_res - (1 / 6)) / (1 - (1 / 6))) * (1 - 0)  # min max normalize
        reward_perf = w_pref * c_perf
        reward_res = w_res * c_res
        reward = -(reward_perf + reward_res)
        return next_state, reward, reward_perf, reward_res


def store_cpu(woker_name):
    global timestamp, change, reset_complete

    cmd = "sudo docker-machine ssh " + woker_name + " docker stats  --no-stream --format \\\"{{ json . }}\\\" "
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
                path = result_dir + name + "_cpu.txt"
                f = open(path, 'a')
                data = str(timestamp) + ' '
                # for d in state_u:
                data = data + str(cpu) + ' ' + '\n'
                f.write(data)
                f.close()


def store_error_count(error):
    # Write the string to a text file
    path = result_dir + "error.txt"
    f = open(path, 'a')
    data = str(error) + '\n'
    f.write(data)


def store_trajectory(service_name, step, s, a, r, r_perf, r_res, s_, done):
    path = result_dir + service_name + "_trajectory.txt"
    tmp_s = list(s)
    tmp_s_ = list(s_)
    f = open(path, 'a')
    data = str(step) + ' ' + str(tmp_s) + ' ' + str(a) + ' ' + str(r) + ' ' + str(r_perf) + ' ' + str(r_res) + ' ' + str(tmp_s_) + ' ' + str(done) + '\n'
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


def post_url(url, rate):

    with ThreadPoolExecutor(max_workers=rate) as executor:

        results = []
        for i in range(rate):
            results.append(executor.submit(post, url))
            time.sleep(1/rate)  # send requests every 1 / rate s

        for result in as_completed(results):
            response, response_time = result.result()
            # # print(type(response.status_code), response_time)
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
            # print('timestamp: ', timestamp)
            event_mn1.clear()  # set flag to false
            event_mn2.clear()
            if ((timestamp) % monitor_period) == 0 and timestamp!=0 :  # every 60s scaling
                event_timestamp_control.set()
                print("wait mn1 mn2 step and service scaling ...")
                event_mn1.wait()  # if flag == false : wait, else if flag == True: continue
                event_mn2.wait()
                change = 0
                print("Start Requesting ...")
            event_timestamp_control.clear()
            # exp = np.random.exponential(scale=1 / i, size=i)
            url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/"
            try:
                post_url(url, i)
            except:
                print("error")
                error += 1
            timestamp += 1


    send_finish = 1
    store_error_count(error)

def agent_threshold(event, service_name):
    global T_max, change, send_finish, replica1, cpus1
    global timestamp
    done = False

    env = Env(service_name)
    step = 1
    state = env.reset()
    while True:
        print(timestamp)
        if timestamp == (monitor_period-5):
            # state[1] = (env.get_cpu_utilization() / 100 / env.cpus)
            state[1] = (env.get_cpu_utilization_from_data() / 100 / env.cpus)
            response_time_list = []
            for i in range(5):
                time.sleep(1)
                response_time_list.append(env.get_response_time())
            mean_response_time = statistics.mean(response_time_list)
            mean_response_time = mean_response_time * 1000
            Rt = mean_response_time
            state[3] = Rt
            break
    print("service name:", env.service_name, "initial state:", state)

    # action: +1 scale out -1 scale in
    while True:
        if timestamp == 0:
            done = False
        event_timestamp_control.wait()
        if (((timestamp) % monitor_period) == 0) and (not done) and timestamp!=0:
            if timestamp == (simulation_time):
                done = True
            else:
                done = False
            # get state
            cpu_utilization = state[1]
            if cpu_utilization >= 0.8:
                action = "+1"
            elif cpu_utilization <= 0.2:
                action = "-1"
            else:
                action = "0"

            next_state, reward, reward_perf, reward_res = env.step(action, event, done)
            print("service name:", env.service_name, "action: ", action, " step: ", step, " next_state: ",
                                     next_state, " reward: ", reward, " done: ", done)
            store_trajectory(env.service_name, step, state, action, reward, reward_perf, reward_res, next_state, done)

            state = next_state
            step += 1
            # event_timestamp_control.clear()
        if done:
            break



t1 = threading.Thread(target=send_request, args=(request_num, total_episodes, ))
t2 = threading.Thread(target=store_cpu, args=('worker',))
t3 = threading.Thread(target=store_cpu, args=('worker1',))
t4 = threading.Thread(target=agent_threshold, args=(event_mn1, 'app_mn1', ))
t5 = threading.Thread(target=agent_threshold, args=(event_mn2, 'app_mn2', ))

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
