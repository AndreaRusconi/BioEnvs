import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))))
os.sys.path.insert(0, parentdir)
print(parentdir)


from stable_baselines import HER,  TD3

from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from BioloidEnviornmentHER_fixed import bioEnv


# -j

action_space = 6
# -p
fixed = False
# -g
gamma = 0.9
# -b
batch_size = 256
# -m
memory_limit = 1000000
# -t
timesteps = 2000000

discreteAction = 0
rend = False

env = bioEnv()

goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
# Wrap the model
model = HER.load("models/TD3/curriculum/best_model_part_11_10g_TRUE.pkl", env=env)

obs = env.reset()
i = 0 
for _ in range(10000):
    i +=1
    action, _ = model.predict(obs)
    print(action)
    obs, reward, done, _ = env.step(action)
    if done:
        print(str(i) + " " + str(done))
        obs = env.reset()
