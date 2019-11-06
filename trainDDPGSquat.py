#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))))
os.sys.path.insert(0, parentdir)
print(parentdir)


from BioloidEnviornment import bioEnv
from stable_baselines import logger
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from termcolor import colored

import datetime
import pybullet_data
import numpy as np
import time
import math as m
import gym
import sys, getopt

def main(argv):

    # -j
    numControlledJoints = 12
    # -p
    fixed = False
    # -o
    normalize_observations = False
    # -g
    gamma = 0.9
    # -b
    batch_size = 16
    # -m
    memory_limit = 1000000
    # -r
    normalize_returns = True
    # -t
    timesteps = 10000000

    policy_name = "Reaching_policy"

    # COMMAND LINE PARAMS MANAGEMENT:
    try:
        opts, args = getopt.getopt(argv,"hj:p:g:b:m:r:o:t:n:",["j=","p=","g=","b=","m=","r=","o=","t=","n="])
    except getopt.GetoptError:
        print ('test.py -t <timesteps> -j <numJoints> -p <fixedPoseObject> -g <gamma> -b <batchsize> -m <memory_limit> -r <norm_ret> -o <norm_obs> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('------------------ Default values:')
            print('train.py -t <timesteps: 10000000> -j <numJoints: 12> -p <fixedPoseObject: False> -n <policy_name:"reaching_policy"> -g <gamma: 0.9> -b <batch_size: 16> -m <memory_limit: 1000000> -r <norm_ret: True> -o <norm_obs: False> ')
            print('------------------')
            return 0
            sys.exit()
        elif opt in ("-j", "--j"):
            if(numControlledJoints >18):
                print("check dim state")
                return 0
            else:
                numControlledJoints = int(arg)
        elif opt in ("-p", "--p"):
            fixed = bool(arg)
        elif opt in ("-g", "--g"):
            gamma = float(arg)
        elif opt in ("-o", "--o"):
            normalize_observations = bool(arg)
        elif opt in ("-b", "--b"):
            batch_size = int(arg)
        elif opt in ("-m", "--m"):
            memory_limit = int(arg)
        elif opt in ("-r", "--r"):
            normalize_returns = bool(arg)
        elif opt in ("-t", "--t"):
            timesteps = int(arg)
        elif opt in ("-n","--n"):
            policy_name = str(arg)




    discreteAction = 0
    rend = False
    bioId = bioEnv()
    n_actions = bioId.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))


    bioId = DummyVecEnv([lambda: bioId])

    model = DDPG(LnMlpPolicy, bioId,normalize_observations = normalize_observations, gamma=gamma,batch_size=batch_size,
                    memory_limit=memory_limit, normalize_returns = normalize_returns, verbose=1, param_noise=param_noise,
                    action_noise=action_noise, tensorboard_log="../pybullet_logs/bioenv_ddpg/", reward_scale = 1)

    print(colored("-----Timesteps:","red"))
    print(colored(timesteps,"red"))
    print(colored("-----Number Joints Controlled:","red"))
    print(colored(numControlledJoints,"red"))
    print(colored("-----Object Position Fixed:","red"))
    print(colored(fixed,"red"))
    print(colored("-----Policy Name:","red"))
    print(colored(policy_name,"red"))
    print(colored("------","red"))
    print(colored("Launch the script with -h for further info","red"))

    model.learn(total_timesteps=timesteps)

 
    model.save("bioEnv_"+ policy_name)

    del model # remove to demonstrate saving and loading

if __name__ == '__main__':
    main(sys.argv[1:])
