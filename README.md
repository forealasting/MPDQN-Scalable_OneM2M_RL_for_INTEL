
## Experimental procedure
![](https://hackmd.io/_uploads/SkHhAqZqn.png)

## Run Code

1. Conda Environment
```bash=bb
conda create -n onem2m python=3.10 
conda activate onem2m
pip install matplotlib
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. Git clone
```bash=
git clone https://github.com/forealasting/Scalable_OneM2M_RL.git
```

3. Set ip
```bash=
sudo docker-machine ls
```
![](https://hackmd.io/_uploads/ryfbxYTPh.png)
![](https://hackmd.io/_uploads/ryJ4ltpv3.png)


4. Choose result directory and traffic 

![](https://hackmd.io/_uploads/HyFUJjZ93.png)


5. Run code
```bash=
python MPDQN.py
```

## Parameter Configuration

### IP Configuration

Modify the following IP addresses in the Python code if your network configuration changes:

```python
ip = "192.168.99.104"  # Update with the new IP address
ip1 = "192.168.99.105"  # Update with the new IP address
```

### Traffic and Request Configuration

Adjust the data rate and traffic parameters in the Python code:

```python
data_rate = 160     # Adjust the data rate as needed
use_tm = 0           # Set to 1 if using traffic management, 0 otherwise
tm_path = 'request/request25.txt'  # Update the traffic path if necessary
result_dir = "./mpdqn_result/result_load_160/result6/"
```

### Simulation Configuration

Configure the simulation parameters in the Python code:

```python
total_episodes = 12   # Total training episodes, adjust as needed
if if_test:
    total_episodes = 1  # Total testing episodes, adjust as needed
monitor_period = 30     # Adjust the monitor period as needed
simulation_time = 3600  # Adjust the simulation time as needed
```

### Manual Action Configuration

If you want to manually set actions for evaluation or debugging, adjust the following settings in the Python code:

```python
manual_action = 0  # Set to 0 if not using manual actions

# Manual action settings for replica 1
manual_action_replica1 = 1
manual_action_cpus1 = 0.8

# Manual action settings for replica 2
manual_action_replica2 = 1
manual_action_cpus2 = 0.8
```

### Learning Parameters

#### **multipass = True -> MPDQN**
#### **multipass = False -> PDQN**

```python
multipass = True  # Set to False for PDQN, True for MPDQN

epsilon_steps = 330  # Adjust the epsilon exploration steps
epsilon_initial = 1   # Set the initial epsilon value
epsilon_final = 0.01  # Set the final epsilon value

learning_rate_actor_param = 0.001
learning_rate_actor = 0.01

tau_actor_param = 0.01
tau_actor = 0.1

gamma = 0.9
replay_memory_size = 960
batch_size = 16
initial_memory_threshold = 16
use_ornstein_noise = False
layers = [64,]
seed = 7

clip_grad = 0  
action_input_layer = 0
```

save the settings in the `settings` dictionary in the Python code:

```python
# Define settings dictionary
settings = {
    # ... (settings)
}
```

### Result Directory

Check and create the result directory :
```python
if os.path.exists(result_dir):
    print("Deleting existing result directory...")
    raise SystemExit  # end process

# Build result directory
os.mkdir(result_dir)
```

### Reference Github
Link : https://github.com/cycraig/MP-DQN

