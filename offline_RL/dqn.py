#  troch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import requests
import time
import threading
import subprocess
import json
import numpy as np
import random
import statistics
from typing import Dict, List, Tuple
import os
import datetime
import concurrent.futures
import math
print(datetime.datetime.now())

# request rate r
data_rate = 50      # if not use_tm
use_tm = 1 # if use_tm
result_dir = "./dqn_database/database1/"

## initial
request_num = []
# timestamp    : 0, 1, 2, 31, ..., 61, ..., 3601
# learning step:          0,  ..., 1,     , 120
#
simulation_time = 3602  # 300 s  # 0 ~ 3601:  3600
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
Rmax_mn1 = 30
Rmax_mn2 = 20


## Learning parameter
# S ={k, u , c, r}
# k (replica): 1 ~ 3                          actual value : same
# u (cpu utilization) : 0.0, 0.1 0.2 ...1     actual value : 0 ~ 100
# c (used cpus) : 0.1 0.2 ... 1               actual value : same
# action_space = ['-r', -1, 0, 1, 'r']
total_episodes = 8       # Total episodes
learning_rate = 0.01          # Learning rate
# Exploration parameters
gamma = 0.9                 # Discounting rate
max_epsilon = 1
min_epsilon = 0
epsilon_decay = 1/840
memory_size = 100
batch_size = 8
target_update = 100

seed = 9
torch.manual_seed(seed)
np.random.seed(seed)


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
data += 'memory_size ' + str(memory_size) + '\n'
data += 'batch_size ' + str(batch_size) + '\n'
data += 'loss function ' + "smooth l1 loss" + '\n'
data += 'target_update ' + str(target_update) + '\n'
f.write(data)
f.close()


## 7/8 stage
stage = ["RFID_Container_for_stage0", "RFID_Container_for_stage1", "Liquid_Level_Container", "RFID_Container_for_stage2",
         "Color_Container", "RFID_Container_for_stage3", "Contrast_Data_Container", "RFID_Container_for_stage4"]

if use_tm:
    f = open('request/request12.txt')

    for line in f:
        if len(request_num) < request_n:

            request_num.append(int(float(line)))
else:
    request_num = [data_rate for i in range(simulation_time)]


print("request_num:: ", len(request_num), "simulation_time:: ", simulation_time)


class Env:

    def __init__(self, service_name="app_mn1"):

        self.service_name = service_name
        self.cpus = 0.5
        self.replica = 1
        self.cpu_utilization = 0.0
        self.state_space = [1, 0.0, 0.5, 10]
        self.n_state = len(self.state_space)
        self.action_space = ['-r', '-1', '0', '1', 'r']
        self.n_actions = len(self.action_space)

        # Need modify ip if container name change
        self.url_list = ["http://" + ip + ":666/~/mn-cse/mn-name/AE1/RFID_Container_for_stage4",
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
            time.sleep(5)  # wait app_mn1 service start
        time.sleep(30)  # wait service start

        if not done:
            # print(self.service_name, "_done: ", done)
            # print(self.service_name, "_step complete")
            event.set()

        response_time_list = []
        time.sleep(20)
        for i in range(5):
            # time.sleep(1)
            response_time_list.append(self.get_response_time())

        if done:
            # print(self.service_name, "_done: ", done)
            time.sleep(10)
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
        next_state.append(request_num[timestamp])
        # next_state.append(req)

        # cost function
        w_pref = 0.5
        w_res = 0.5
        c_perf = 0 + ((c_perf - math.exp(-50/t_max)) / (1 - math.exp(-50/t_max))) * (1 - 0)
        c_res = 0 + ((c_res - (1 / 6)) / (1 - (1 / 6))) * (1 - 0)
        reward_perf = w_pref * c_perf
        reward_res = w_res * c_res
        reward = -(reward_perf + reward_res)
        return next_state, reward, reward_perf, reward_res


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)


class DQNAgent:
    def __init__(
            self,
            env,  # need change
            memory_size: int,
            batch_size: int,
            target_update: int,
            epsilon_decay: float,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            gamma: float = 0.9,
    ):

        # obs_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.n
        obs_dim = 4  # S = {k, u , c}  # S = {k, u , c, r}
        action_dim = 5  # 𝐴={−𝑟, −1,  0,  1,  𝑟}

        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.01)

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        available_actions = self.get_available_actions(state)
        action_mask = np.isin(range(5), available_actions)
        selected_action_idx = np.where(action_mask)[0]  # find True index

        # epsilon greedy policy
        if self.epsilon > np.random.random():
            # print("random action")
            selected_action = np.random.choice(selected_action_idx)
        else:
            q_values = self.dqn(torch.FloatTensor(state).to(self.device))
            # print(q_values)
            masked_q_values = torch.where(
                torch.BoolTensor(action_mask).to(self.device),
                q_values,
                torch.tensor(-np.inf).to(self.device)
            )
            selected_action = masked_q_values.argmax()
        selected_action = selected_action.item()
        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray, event,  done: bool) -> Tuple[np.ndarray, np.float64, np.float64, np.float64]:
        """Take an action and return the response of the env."""
        # next_state, reward, done, _ = self.env.step(action)
        next_state, reward, reward_perf, reward_res = self.env.step(action, event,  done)
        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, reward_perf, reward_res

    def update_model(self) -> float:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, episodes: int, event,  plotting_interval: int = 200):
        global timestamp, simulation_time, change, send_finish

        """Train the agent."""
        self.is_test = False
        update_cnt = 0
        epsilons = []
        reward = 0
        init_state = [1, 0.0, 0.5, 10]
        init_state = np.array(init_state, dtype=float)
        step = 0
        for episode in range(episodes):
            state = init_state
            done = False
            losses = []
            rewards = []

            # if self.env.service_name == "app_mn1":
            #     print("service name:", self.env.service_name, " episode:", episode)
            event_timestamp_Ccontrol.wait()
            print("service name:", self.env.service_name, " episode:", episode)
            while True:
                # if training is ready
                if len(self.memory) >= self.batch_size:
                    # print("training")
                    loss = self.update_model()
                    losses.append(loss)
                    store_loss(self.env.service_name, loss)
                    update_cnt += 1

                    # if hard update is needed
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                if timestamp == 0:
                    done = False
                event_timestamp_Ccontrol.wait()
                if (((timestamp - 1) % 30) == 0) and (not done):
                    action = self.select_action(state)
                    if timestamp == (simulation_time - 1):
                        done = True
                    else:
                        done = False
                    next_state, reward, reward_perf, reward_res = self.step(action, event, done)
                    # if self.env.service_name == "app_mn1":
                    print("service name:", self.env.service_name, "action: ", action, " step: ", step, " next_state: ",
                          next_state, " reward: ", reward, " done: ", done)
                    store_trajectory(self.env.service_name, step, state, action, reward, reward_perf, reward_res, next_state, done)
                    state = next_state

                    # linearly decrease epsilon
                    self.epsilon = max(
                        self.min_epsilon, self.epsilon - (
                                self.max_epsilon - self.min_epsilon
                        ) * self.epsilon_decay
                    )
                    epsilons.append(self.epsilon)

                    step += 1
                    event_timestamp_Ccontrol.clear()

                if done:
                    # state = self.env.reset()
                    print("done")
                    rewards.append(reward)
                    break

            # store_reward(self.env.service_name, avg_rewards)

        torch.save(self.dqn, result_dir + self.env.service_name + '.pt')


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

    def test(self):
        """Test the agent."""
        self.is_test = True

        state = self.env.reset()
        done = False
        reward = 0

        # frames = []

        print("reward: ", reward)
        # self.env.close()



    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        available_actions_batch = [self.get_available_actions(obs) for obs in samples["next_obs"]]
        action_mask = np.array([np.isin(range(5), available_actions) for available_actions in available_actions_batch])

        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(
            next_state
        )
        masked_q_values = torch.where(
            torch.BoolTensor(action_mask).to(self.device),
            next_q_value,
            torch.tensor(-np.inf).to(self.device)
        )
        masked_q_values = masked_q_values.detach().max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * masked_q_values * mask).to(self.device)

        # calculate dqn loss
        # loss = F.mse_loss(curr_q_value, target)
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

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



def dqn(total_episodes, memory_size, batch_size, target_update, epsilon_decay, event, service_name):
    global timestamp, simulation_time, change, RFID, send_finish

    env = Env(service_name)
    agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay, max_epsilon, min_epsilon, gamma)
    agent.train(total_episodes, event)


start_time = time.time()

t1 = threading.Thread(target=send_request, args=(stage, request_num, start_time, total_episodes, ))
t2 = threading.Thread(target=store_cpu, args=(start_time, 'worker',))
t3 = threading.Thread(target=store_cpu, args=(start_time, 'worker1',))
t4 = threading.Thread(target=dqn, args=(total_episodes, memory_size, batch_size, target_update, epsilon_decay, event_mn1, 'app_mn1', ))
t5 = threading.Thread(target=dqn, args=(total_episodes, memory_size, batch_size, target_update, epsilon_decay, event_mn2, 'app_mn2', ))

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

