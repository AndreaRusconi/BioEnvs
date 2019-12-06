import os , inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

from collections import OrderedDict
import gym
from gym import spaces
import pybullet as p
import time
import numpy as np
import copy
import math as m
import pybullet_data

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

#used in save_data_test
test_steps = 0
test_done = False

def goal_distance(goal_a, goal_b):
    return np.linalg.norm(goal_a - goal_b, axis = -1)

largeValObservation = 100

class bioEnv(gym.GoalEnv) :
    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50 }
    def __init__(self, urdfRootPath="../ros-bioloid/src/bioloid_master/urdf/mioloid_robot_head.urdf",
        actionRepeat=1,
        basePosition = [0,0,0.215], 
        baseOrientation = p.getQuaternionFromEuler([0,0,-0.5831853071795866]), 
        numConrolledJoints=6, 
        renders = True,
        max_episode_steps = 400,
        test_phase = False,
        maxSteps = 400,
        dist_delta = 0.03):
        print("init" + urdfRootPath)
        self.urdfRootPath= urdfRootPath
        self.timeStep = 1./240.
        self.terminated = False
        self.basePosition = basePosition
        self.baseOrientation = baseOrientation
        #self.test_phase = test_phase
        self.achieved_goal = []
        self._observation = []
        self._envStepCounter = 0
        self.numControlledJoints = 6
        self.action_dim = 6
        self._maxSteps = maxSteps
        self.max_episode_steps = max_episode_steps
        self.renders =renders
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self.ep_counter = -1
        self._actionRepeat = actionRepeat
        self.freeJointList = [ 14, 15, 16, 21, 22, 23] #[]
        self.targetObservation = [ 0, 0, 0.15,     0.0, 0.0, -0.5831853071795866] 
        self._target_pos_dist_min = 0.02
        self._target_orn_dist_min = 0.1
        

        if self.renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid<0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(2.5,90,-60,[0.52,-0.2,-0.33])
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.totalObservation = self.reset()


        observation_dim = len(self.totalObservation['observation'])

        self.observation_space = spaces.Dict({

            'observation': spaces.Box(-largeValObservation, largeValObservation, shape=(12,), dtype=np.float32),

            #the achieved goal is the position reached 
            'achieved_goal': spaces.Box(-largeValObservation, largeValObservation, shape=(6,), dtype=np.float32),

            #the desired goal is the desired position in space
            'desired_goal': spaces.Box(-largeValObservation, largeValObservation, shape=(6,), dtype=np.float32)

            })

        print(self.observation_space)

        self._action_bound = 1
        action_high = np.array([self._action_bound] * self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype='float32')
            
        self.viewer = None

    def step(self, action):
    	action = [float(i*0.05) for i in action]
    	return self.step2(action)

    def step2(self,action):
       
        for i in range(self._actionRepeat):
            self.applyAction(action)
            p.stepSimulation()
            self._envStepCounter += 1

        self.totalObservation = self.getExtendedObservation()

        reward = self.compute_reward(self.totalObservation['achieved_goal'], self.totalObservation['desired_goal'], None)
        #if the reward is zero done = TRUE
        done = reward == 0
        info = {'is_success': done}
        done = done or self._envStepCounter >= self._maxSteps
        return self.totalObservation, reward, done, info
    

    def render(self, mode="rgb_array", close=False):
        ## TODO Check the behavior of this function
        if mode != "rgb_array":
          return np.array([])

        base_pos,orn = self._p.getBasePositionAndOrientation(self.bioId)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
            #renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array
    
    def reset(self):
        
        #self.freeJointList = []
       
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self.timeStep)
        self._envStepCounter = 0
        self.ep_counter = self.ep_counter + 1

        #loading Plane
        planeId = p.loadURDF("plane.urdf")

        #Load bioloid
        self.bioId = p.loadURDF(self.urdfRootPath, basePosition = self.basePosition , baseOrientation = self.baseOrientation, useFixedBase= False )
        joints_num = p.getNumJoints(self.bioId)
        
        for i in range(joints_num):
            p.resetJointState(self.bioId, i, 0)
            p.setJointMotorControl2(self.bioId, i, p.POSITION_CONTROL,targetPosition=0,targetVelocity=0.0,
            positionGain=0.25, velocityGain=0.4, force=25)

        self._debugGUI()
        p.setGravity(0,0,-9.8)
        #add debug slider
        #target_joint_pos = [0,0,0,      0,0,0,      0,0,1.188,-2.315,1.188,0,       0,0,1.188,-2.315,1.188,0]
        #init_joint_pos =[0,0,0,      0,0,0,      0,0,0,0,0,0,        0,0,0,0,0,0]
        jointIds=[]
        paramIds=[]
        joints_num = p.getNumJoints(self.bioId)

        i=0
        """
        for j in range(26):
            info = p.getJointInfo(self.bioId,j)
            if info[2] == 0 :   
                self.freeJointList.append(j)
        """
        # Let the world run for a bit
       	for _ in range(10):
            p.stepSimulation()

        #self._observation = self.getObservation() #[getlinkState[0](3 campi), getLinkState[1](3 campi), jointPoses (18 campi) ]
        return self.getExtendedObservation()

    def getExtendedObservation(self):

        self._observation = self.getObservation()
        self.achieved_goal = self._observation[0:6]


        return OrderedDict([
            ('observation', np.asarray(self._observation.copy())),
            ('achieved_goal', np.asarray(self.achieved_goal.copy())),
            ('desired_goal', np.asarray(list(self.targetObservation).copy()))
            ])

    
    def getActionDimension(self):
        return self.action_space

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        
        observation = []
        pos , orn = p.getBasePositionAndOrientation(self.bioId)
        euler = p.getEulerFromQuaternion(orn)
        observation.extend(list(pos))
        observation.extend(list(euler)) 

        jointStates = []
        jointPoses = []
        for i in range(len(self.freeJointList)):
            jointStates.append(p.getJointState(self.bioId, self.freeJointList[i]))
        jointPoses = [x[0] for x in jointStates]
        observation.extend(list(jointPoses))

        return observation

    def applyAction(self, action):
    	for a in range(len(self.freeJointList)):
        	curr_motor_pos = p.getJointState(self.bioId, self.freeJointList[a])[0]
        	new_motor_pos = curr_motor_pos + action[a] #supposed to be a delta
        	p.setJointMotorControl2(self.bioId,self.freeJointList[a],p.POSITION_CONTROL,targetPosition=new_motor_pos,targetVelocity=0,positionGain=0.25,velocityGain=0.4,force=25)	
   
    def compute_reward(self,  achieved_goal, desired_goal, info):
        
       	global test_done
        d_pos = goal_distance(np.array(achieved_goal[0:3]), np.array(desired_goal[0:3]))
        d_orn = goal_distance(np.array(achieved_goal[3:6]), np.array(desired_goal[3:6]))

        if ((d_pos <= self._target_pos_dist_min) and (d_orn <= self._target_orn_dist_min)) :
            test_done = True
            return 0
        else:
            return -1

    def _debugGUI(self):
        #TO DO 
        return 0 


  
               
