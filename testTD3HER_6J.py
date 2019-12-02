import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))))
os.sys.path.insert(0, parentdir)
print(parentdir)


from stable_baselines import HER, DQN, SAC, DDPG, TD3

from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from BioloidEnviornmentHER_6J import bioEnv
model_class = DDPG  # works also with SAC and DDPG

# -j

action_space = 6
# -p
fixed = False
# -o
normalize_observations = False
# -g
gamma = 0.9
# -b
batch_size = 256
# -m
memory_limit = 1000000
# -r
normalize_returns = False
# -t
timesteps = 15000000
policy_name = "pushing_policy"
discreteAction = 0
rend = False

env = bioEnv(renders=True)

goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
# Wrap the model
model = HER.load("models/TD3/best_model.pkl", env=env)

obs = env.reset()

for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()