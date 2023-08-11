from locust import HttpUser, task, constant_pacing
import random
import subprocess
from locust.stats import stats_printer, stats_history
from locust.env import Environment
from locust.log import setup_logging
import statistics


error_rate = 0.2

ip = "192.168.99.104"  # app_mn1
ip1 = "192.168.99.105"  # app_mn2
url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/"

# http://192.168.99.128:666/~/mn-cse/mn-name/AE1/
## 8 sensors
sensors = ["RFID_Container_for_stage0", "RFID_Container_for_stage1", "Liquid_Level_Container", "RFID_Container_for_stage2",
         "Color_Container", "RFID_Container_for_stage3", "Contrast_Data_Container", "RFID_Container_for_stage4"]

response_times = []


def reset(r1, r2):
    print("reset envronment...")
    cmd_list = [
        "sudo docker-machine ssh default docker service update --replicas 0 app_mnae1",
        "sudo docker-machine ssh default docker service update --replicas 0 app_mnae2",
        "sudo docker-machine ssh default docker service update --replicas 0 app_mn1",
        "sudo docker-machine ssh default docker service update --replicas 0 app_mn2",
        "sudo docker-machine ssh default docker service update --replicas 1 app_mnae1",
        "sudo docker-machine ssh default docker service update --replicas 1 app_mnae2",
        "sudo docker-machine ssh default docker service update --replicas " + str(r1) + " app_mn1",
        "sudo docker-machine ssh default docker service update --replicas " + str(r2) + " app_mn2",
    ]
    def execute_command(cmd):
        return subprocess.check_output(cmd, shell=True)
    for cmd in cmd_list:
        result = execute_command(cmd)
        print(result)

reset(1,1)

class OneM2MUser(HttpUser):

    wait_time = constant_pacing(1 / 10)  # 每秒生成50個請求
    @task
    def on_start(self):
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
        # 執行POST請求
        response = self.client.post(url1, headers=headers, json=data)

        if response.status_code == 201:
            print("POST successful")
        else:
            print("POST failed")

    @task
    def index(self):
        self.client.get("/")
        self.client.get("/static/assets.js")

    @task
    def about(self):
        self.client.get("/about/")