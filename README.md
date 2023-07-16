# Scalable_OneM2M_RL
																	
## Environment (The environments of the following code are all different)

## DQN(dqn.py)

## PDQN(MPDQN.py) multipass set False

## MPDQN(MPDQN.py) multipass set True

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


4. choose result directory and traffic 

![](https://hackmd.io/_uploads/HyFUJjZ93.png)


5. run code
```bash=
python MPDQN.py
```

