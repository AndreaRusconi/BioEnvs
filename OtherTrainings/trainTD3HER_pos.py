#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))))
os.sys.path.insert(0, parentdir)


from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines.ddpg.policies import LnMlpPolicy


import tensorflow as tf
from stable_baselines.td3.policies import FeedForwardPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
import numpy as np
from BioloidEnviornmentHER_pos import bioEnv


class CustomTD3Policy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(*args, **kwargs,
                                           layers=[128,128,128],
                                           layer_norm=False,
                                           act_fun=tf.nn.relu,
                                           feature_extraction="lnmlp")

best_mean_reward, n_steps = -np.inf, 0
log_dir="../pybullet_logs/bioEnv_TD3/"
log_dir_policy = "models/TD3/"



def callback(_locals, _globals):

    global n_steps, best_mean_reward, log_dir
    # Print stats every 1000 calls
    if (n_steps) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir_policy + 'best_model.pkl')
    n_steps += 1
    return True

def main(load_policy=False):
    global log_dir
    model_class = TD3  # works also with SAC and DDPG
    action_space = 12
    normalize_observations = False
    gamma = 0.9
    memory_limit = 1000000
    normalize_returns = True
    timesteps = 5000000
    discreteAction = 0
    rend = False

    env = bioEnv()
  
    env = Monitor(env, log_dir, allow_early_resets=True)

    goal_selection_strategy = 'future'
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    # Wrap the model

    model = HER(CustomTD3Policy, env, model_class,n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                verbose=1,tensorboard_log="../pybullet_logs/bioEnv_TD3", buffer_size=1000000,batch_size= 256,
                random_exploration=0.3, action_noise=action_noise)
    
    if (load_policy):
        model = HER.load("models/TD3/best_model_1.pkl", env=env, n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        tensorboard_log="../pybullet_logs/bioEnv_TD3",
        buffer_size=1000000,batch_size=256,random_exploration=0.3, action_noise=action_noise)
    
    model.learn(timesteps,log_interval=100, callback = callback)
   
    model.save("policy_TD3_new")

if __name__ == "__main__":
    main()