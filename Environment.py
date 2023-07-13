import statistics
import subprocess
import time
import requests
import threading
# service_name_list = ["app_mn1", "app_mn2", "app_mnae1", "app_mnae2"]
service_name_list = ["app_mn1", "app_mn2"]
# Kmax = 3  #  max number of replicas
# u = 11    #  divide cpu utilization into 11 degrees
# c = 10    #  divide cpus into 10 degrees
RFID = 0
timestamp = 0  # timestamp
# send_finish = 0
event = threading.Event()
# Need modify ip if ip change
ip = "192.168.99.121"
ip1 = "192.168.99.122"

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
        global RFID
        path1 = "result/" + self.service_name + "_response.txt"

        f1 = open(path1, 'a')

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
        start = time.time()
        response = requests.post(url, headers=headers, json=data)
        end = time.time()
        response_time = end - start
        data1 = str(timestamp) + ' ' + str(response_time) + ' ' + str(self.cpus) + ' ' + str(self.replica) + '\n'
        f1.write(data1)
        f1.close()
        return response_time

    def get_cpu_utilization(self):
        path = "result/" + self.service_name + '_cpu.txt'
        try:
            f = open(path, "r")
            cpu = []
            time = []
            for line in f:
                s = line.split(' ')
                time.append(float(s[0]))
                cpu.append(float(s[2]))

            last_avg_cpu = sum(cpu[-3:])/len(cpu[-3:])
            f.close()

            return last_avg_cpu
        except:
            print("self.service_name:: ", self.service_name)
            print('cant open')

    def discretize_cpu_value(self, value):
        return int(round(value / 10))

    def step(self, action_index, event, done):
        global timestamp, send_finish, RFID, change, simulation_time

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

        time.sleep(30)
        if not done:
            event.set()
        print(self.service_name, "_done:: ", done)
        response_time_list = []
        for i in range(5):
            time.sleep(3)
            response_time_list.append(self.get_response_time())

        event.set()  # if done and after get_response_time
        # avg_response_time = sum(response_time_list)/len(response_time_list)
        median_response_time = statistics.median(response_time_list)
        median_response_time = median_response_time*1000  # 0.05s -> 50ms
        if median_response_time >= 50:
            Rt = 50
        else:
            Rt = median_response_time
        if self.service_name == "app_mn1":
            t_max = 25
        elif self.service_name == "app_mn2":
            t_max = 20
        else:
            t_max = 5

        if median_response_time < t_max:
            c_perf = 0
        else:
            tmp_d = 1.4 ** (50 / t_max)
            tmp_n = 1.4 ** (Rt / t_max)
            c_perf = tmp_n / tmp_d

        c_res = (self.replica*self.cpus)/3   # replica*self.cpus / Kmax
        next_state = []
        # k, u, c # r
        self.cpu_utilization = self.get_cpu_utilization()
        u = self.discretize_cpu_value(self.cpu_utilization)
        next_state.append(self.replica)
        next_state.append(u/10)
        next_state.append(self.cpus)
        # state.append(req)
        w_pref = 0.5
        w_res = 0.5
        reward = -(w_pref * c_perf + w_res * c_res)
        # print("step_over_next_state: ", next_state)
        return next_state, reward

