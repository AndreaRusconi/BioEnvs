import os , inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym
from gym import spaces
import pybullet as p
import time
import numpy as np
import copy
import math as m
import pybullet_data

def goal_distance(goal_a, goal_b):
    return np.linalg.norm(goal_a - goal_b, axis = -1)


                  

largeValObservation = 100
class bioEnv(gym.Env) :
    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50 }

    def __init__(self, urdfRootPath="../ros-bioloid/src/bioloid_master/urdf/mioloid_robot_head.urdf",actionRepeat=1, basePosition = [0,0,0.22], baseOrientation = p.getQuaternionFromEuler([0,0,5.8]), numConrolledJoints=18, renders= False):
        print("init" + urdfRootPath)
        self.urdfRootPath= urdfRootPath
        self.timeStep = 1./240.
        self.useOrientation = 1
        self.useSimulation = 1
        self.terminated = False
        self.basePosition = basePosition
        self.baseOrientation = baseOrientation
        self.i =0 
        self.achieved_goal = []
        self._envStepCounter = 0
        self._observation = []
        self.numJoints = 26
        self.numControlledJoints = 18
        self.action_dim = 12
        self._maxSteps = 1000
        self.renders =renders
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self.targetObservation=[-0.00022756908581942603 , -0.000200384263379346 , 0.00010279144557757175 , 0.00015010634498172459 , 0.0002050663891038947 , -0.00024082488576043486 ,  1.1880585171461655 , -2.31561388612597, 1.1892599041779879 , 1.1880174349839125 , -2.3143761387021726  , 1.1886576104218778 ]#[-0.00022756908581942603 4, -0.000200384263379346 5, 0.00010279144557757175 6, 0.00015010634498172459 8, 0.0002050663891038947 9, -0.00024082488576043486 10 , 0.0003120330558368287 12, 0.00028589470767586384 13, 1.1880585171461655 14, -2.31561388612597 15, 1.1892599041779879 16, 0.0026689512345122145 17, 0.0002063148112800855 19, 0.000124681430460826542 0, 1.1880174349839125 21, -2.3143761387021726 22 , 1.1886576104218778 23, -0.0014707933869243738 24]
        self._actionRepeat = actionRepeat
        self.freeJointList = [4, 5, 6, 8, 9, 10, 14, 15, 16, 21, 22, 23]
        #self.targetObservation = [-0.00790527938749156, 0.032147963452107825, 0.23272512105554582, -1.5723172039419893, -0.06225515159135731, 0.009696014494046348, -0.00022756908581942603, -0.000200384263379346, 0.00010279144557757175, 0.00015010634498172459, 0.0002050663891038947, -0.00024082488576043486, 0.0003120330558368287, 0.00028589470767586384, 1.1880585171461655, -2.31561388612597, 1.1892599041779879, 0.0026689512345122145, 0.0002063148112800855, 0.00012468143046082654, 1.1880174349839125, -2.3143761387021726, 1.1886576104218778, -0.0014707933869243738]
        self._target_dist_min = 0.2
        self.target_joint_pos = [0,0,0,      0,0,0,      0,0,1.188,-2.315,1.188,0,       0,0,1.188,-2.315,1.188,0]
       	self.viewer = None

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.reset()

        observationDim = len(self._observation)
        observation_high = np.array([largeValObservation] * observationDim)
        self.observation_space = spaces.Box(-observation_high, observation_high, dtype='float32')

       
        #self.action_dim = 2 #self._panda.getActionDimension()
        self._action_bound = 1
        action_high = np.array([self._action_bound] * self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype='float32')

    def step(self, action):
    	action = [float(i*0.05) for i in action]
    	return self.step2(action)

    def step2(self,action):
      
        
        for i in range(self._actionRepeat):
            self.applyAction(action)
            p.stepSimulation()

            if self._termination():
                break
            self._envStepCounter += 1

        
        self._observation = self.getObservation()

        reward = self._compute_reward()

        done = self._termination()

        return np.array(self._observation), np.array([reward]), np.array(done), {}

    def reset(self):
       
        #self.freeJointList = []
        self.terminated = False
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self.timeStep)
        self._envStepCounter = 0
    

        #loading Plane
        planeId = p.loadURDF("plane.urdf")

        #Load bioloid
        self.bioId = p.loadURDF(self.urdfRootPath, basePosition = self.basePosition , baseOrientation = self.baseOrientation )
        for i in range(self.numJoints):
            p.resetJointState(self.bioId, i, 0)
            p.setJointMotorControl2(self.bioId, i, p.POSITION_CONTROL,targetPosition=0,targetVelocity=0.0,
            positionGain=0.25, velocityGain=0.75, force=50)
        self._debugGUI()
        p.setGravity(0,0,-9.8)
        #add debug slider
        target_joint_pos = [0,0,0,      0,0,0,      0,0,1.188,-2.315,1.188,0,       0,0,1.188,-2.315,1.188,0]
        init_joint_pos =[0,0,0,      0,0,0,      0,0,0,0,0,0,        0,0,0,0,0,0]
        jointIds=[]
        paramIds=[]
        joints_num = p.getNumJoints(self.bioId)

        i=0
        """
        for j in range(joints_num):
            info = p.getJointInfo(self.bioId,j)
            print(info)
            if info[2] == 0 :   
                self.freeJointList.append(j)
        """
        #print(self.freeJointList)
        # Let the world run for a bit
       	for _ in range(10000):
            p.stepSimulation()
        
        	
        self._observation = self.getObservation() #[pos[0](3 campi), euler(3 campi), jointPoses (18 campi) ]
        return np.array(self._observation)
    
    def getActionDimension(self):
        return self.action_space

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
      
        observation = []
        pos , orn = p.getBasePositionAndOrientation(self.bioId)
        euler = p.getEulerFromQuaternion(orn)
        #observation.extend(list(pos))
        #observation.extend(list(euler)) #roll, pitch, yaw
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
            p.setJointMotorControl2(self.bioId,self.freeJointList[a],p.POSITION_CONTROL,targetPosition=new_motor_pos,targetVelocity=0,positionGain=0.25,velocityGain=0.75,force=100)
    
    def _termination(self):
        d = goal_distance(np.array(self._observation), np.array(self.targetObservation))

        if d <= self._target_dist_min:
            self.terminated = True


        if (self.terminated or self._envStepCounter > self._maxSteps):
            self._observation = self.getObservation()
            return [True]

        return [False]
   
    def _compute_reward(self):
        
        reward = np.float(32.0)
        
        d =goal_distance(np.array(self._observation), np.array(self.targetObservation))

        reward = -d
        if d <= self._target_dist_min:
            reward = np.float32(1000.0) + (100 - d*80)
        return reward


    def _debugGUI(self):
        #TO DO 
        return 0 


  
               
