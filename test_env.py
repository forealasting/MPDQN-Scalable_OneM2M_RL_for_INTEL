import requests
import concurrent.futures
import time
import threading
import subprocess
import json
import numpy as np
import random
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# define result path
# result_dir = "./static_result/0521/request_50/result1/"
result_dir = "./static_result/result50/"
# request rate r
data_rate = 70  # use static request rate
use_tm = 0  # use dynamic traffic
error_rate = 0.2   # 0.2

## initial
request_num = []
simulation_time = 100  # 300 s  # 3600s
cpus1 = 0.8
replica1 = 3

cpus2 = 1
replica2 = 1

request_n = simulation_time
change = 0   # 1 if take action / 0 if init or after taking action
send_finish = 0  # 1 : finish
timestamp = 0    # time record
RFID = 0   # For different RFID data

ip = "192.168.99.124"  # app_mn1
ip1 = "192.168.99.125"  # app_mn2
# url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/"


## Sensor i for every sensors
sensors = ["RFID_Container_for_stage0", "RFID_Container_for_stage1", "Liquid_Level_Container", "RFID_Container_for_stage2",
         "Color_Container", "RFID_Container_for_stage3", "Contrast_Data_Container", "RFID_Container_for_stage4"]


# check result directory
if os.path.exists(result_dir):
    print("Deleting existing result directory...")
    raise SystemExit  # end process

# build dir
os.mkdir(result_dir)

# store setting
path = result_dir + "setting.txt"
f = open(path, 'a')
data = 'data_rate: ' + str(data_rate) + '\n'
data += 'use_tm: ' + str(use_tm) + '\n'
data += 'simulation_time ' + str(simulation_time) + '\n'
data += 'cpus: ' + str(cpus1) + '\n'
data += 'replica: ' + str(replica1) + '\n'
data += 'cpus: ' + str(cpus2) + '\n'
data += 'replica: ' + str(replica2) + '\n'
f.write(data)
f.close()

if use_tm:
    #   Modify the workload path if it is different
    f = open('request/request14.txt')

    for line in f:
        if len(request_num) < request_n:

            request_num.append(int(float(line)))
else:
    request_num = [data_rate for i in range(simulation_time)]

print('request_num:: ', len(request_num))


def post_url(url, RFID):

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

    try:
        # response = requests.post(url, headers=headers, json=data, timeout=0.05)
        response = requests.post(url, headers=headers, json=data, timeout=0.05)
        response = str(response.status_code)
    except requests.exceptions.Timeout:
        response = 'timeout'



def store_cpu(worker_name):
    global timestamp, change

    cmd = "sudo docker-machine ssh " + worker_name + " docker stats --all --no-stream --format \\\"{{ json . }}\\\" "
    while True:
        if send_finish == 1:
            break
        if change == 0:
            returned_text = subprocess.check_output(cmd, shell=True)
            my_data = returned_text.decode('utf8')
            # print(my_data.find("CPUPerc"))
            my_data = my_data.split("}")
            cpu_list = []
            for i in range(len(my_data) - 1):
                # print(my_data[i]+"}")
                my_json = json.loads(my_data[i] + "}")
                name = my_json['Name'].split(".")[0]
                cpu = my_json['CPUPerc'].split("%")[0]
                if float(cpu) > 0 :
                    cpu_list.append(float(cpu))
                    path = result_dir + name + "_cpu.txt"
                    f = open(path, 'a')
                    data = str(timestamp) + ' '
                    data = data + str(cpu) + ' ' + '\n'
                    f.write(data)
                    f.close()



def store_rt(timestamp, response, rt):
    path = result_dir + "app_mn1_response.txt"
    f = open(path, 'a')
    for i in range(len(timestamp)):
        data = str(timestamp[i]) + ' ' + str(response[i]) + ' ' + str(rt[i]) + '\n'
        f.write(data)
    f.close()



# sned request to app_mn2 # app_mnae1 app_mnae2
def store_rt2():
    global timestamp, send_finish, change

    path1 = result_dir + "/app_mn2_response.txt"

    while True:
        if change == 0:
            f1 = open(path1, 'a')

            RFID1 = random.randint(0, 1000000)

            headers = {"X-M2M-Origin": "admin:admin", "Content-Type": "application/json;ty=4"}
            data = {
                "m2m:cin": {
                    "con": "true",
                    "cnf": "application/json",
                    "lbl": "req",
                    "rn": str(RFID1),
                }
            }

            # URL 1
            url = "http://" + ip1 + ":777/~/mn-cse/mn-name/AE2/Control_Command_Container"

            try:
                s_time = time.time()
                response = requests.post(url, headers=headers, json=data, timeout=0.05)
                response1 = str(response.status_code)
                response_time1 = time.time() - s_time

            except requests.exceptions.Timeout:
                response1 = 'timeout'
                response_time1 = 0.05
            data1 = str(timestamp) + ' ' + str(response1) + ' ' + str(response_time1) + '\n'
            f1.write(data1)
            time.sleep(1)

            if send_finish == 1:
                f1.close()
                break

def store_rt1():
    global timestamp, send_finish, change

    path1 = result_dir + "/app_mn1_response.txt"

    while True:
        if change == 0:
            f1 = open(path1, 'a')

            RFID1 = random.randint(0, 1000000)

            headers = {"X-M2M-Origin": "admin:admin", "Content-Type": "application/json;ty=4"}
            data = {
                "m2m:cin": {
                    "con": "true",
                    "cnf": "application/json",
                    "lbl": "req",
                    "rn": str(RFID1),
                }
            }

            # URL 1
            url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/RFID_Container_for_stage4"

            try:
                s_time = time.time()
                response = requests.post(url, headers=headers, json=data, timeout=0.05)
                response1 = str(response.status_code)
                response_time1 = time.time() - s_time

            except requests.exceptions.Timeout:
                response1 = 'timeout'
                response_time1 = 0.05
            data1 = str(timestamp) + ' ' + str(response1) + ' ' + str(response_time1) + '\n'
            f1.write(data1)
            time.sleep(1)

            if send_finish == 1:
                f1.close()
                break


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

def reset():
    cmd_list = [
        "sudo docker-machine ssh default docker service update --replicas 0 app_mn1",
        "sudo docker-machine ssh default docker service update --replicas 0 app_mn2",
        "sudo docker-machine ssh default docker service update --replicas 1 app_mn1",
        "sudo docker-machine ssh default docker service update --replicas 1 app_mn2",
        "sudo docker-machine ssh default docker service update --limit-cpu 1 app_mn1",
        "sudo docker-machine ssh default docker service update --limit-cpu 1 app_mn2"
    ]
    def execute_command(cmd):
        return subprocess.check_output(cmd, shell=True)

    with ThreadPoolExecutor(max_workers=len(cmd_list)) as executor:
        results = list(executor.map(execute_command, cmd_list))
def send_request(sensors, request_num):
    global change, send_finish
    global timestamp, use_tm, RFID
    reset()
    time.sleep(70)
    error = 0

    for i in request_num:
        #print(timestamp)
        #exp = np.random.exponential(scale=1 / i, size=i)
        url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/"
        try:
            post_url(url, i)
        except:
            print("error")
            error += 1

        timestamp += 1
    send_finish = 1


t1 = threading.Thread(target=send_request, args=(sensors, request_num, ))
t2 = threading.Thread(target=store_cpu, args=('worker',))
t3 = threading.Thread(target=store_cpu, args=('worker1',))
t4 = threading.Thread(target=store_rt2)
t5 = threading.Thread(target=store_rt1)

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

