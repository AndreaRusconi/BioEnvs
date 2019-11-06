import pybullet as p
import time
import math as m
import pybullet_data
from BioloidEnviornment import bioEnv
import numpy as np 

#bioEnv = bioEnv()

def goal_distance(goal_a, goal_b):
    return np.linalg.norm(goal_a - goal_b, axis = -1)

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
 
print(pybullet_data.getDataPath())
#Set gravity 
p.setGravity(0,0,-9.8)


p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf")
basePosition = [0,0,0.23]

baseOrientation = p.getQuaternionFromEuler([0,0,5.8])

bioId = p.loadURDF("../ros-bioloid/src/bioloid_master/urdf/mioloid_robot_head.urdf", basePosition ,baseOrientation ,useFixedBase = False)
print("Bioloid id:"+ str(bioId))

#add debug slider
init_pos = [-0.00022756908581942603 , -0.000200384263379346 , 0.00010279144557757175 , 0.00015010634498172459 , 0.0002050663891038947 , -0.00024082488576043486 ,  1.1880585171461655 , -2.31561388612597, 1.1892599041779879 , 1.1880174349839125 , -2.3143761387021726  , 1.1886576104218778 ]
jointIds=[]
paramIds=[]
jointName=[]
freeJointList = [4, 5, 6, 8, 9, 10, 14, 15, 16, 21, 22, 23]
joints_num = p.getNumJoints(bioId)

print("len init_pos ",len(init_pos))
print("Number of joints:"+ str(joints_num))

for k in range(joints_num):
	a = p.getJointInfo(bioId,k)
	print(a)

for j in range(len(freeJointList)):
    info = p.getJointInfo(bioId,freeJointList[j])
    jointName=info[1]
    paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), info[8], info[9], init_pos[j]))
	
while True: 
	pos = []
	for i in range(len(freeJointList)):
		p.setJointMotorControl2(bioId,freeJointList[i], p.POSITION_CONTROL,targetPosition=p.readUserDebugParameter(i),targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)
	for k in range (len(freeJointList)):
		info = p.getJointState(bioId, freeJointList[k])
		pos.append(info[0])
	
	p.stepSimulation()
	d = goal_distance(np.array(pos), np.array(init_pos))
	print(d)
	time.sleep(0.01)


   		

bioloidPos, bioloidOrn = p.getBasePositionAndOrientation(bioId)
print(bioloidPos,bioloidOrn)
p.disconnect()
