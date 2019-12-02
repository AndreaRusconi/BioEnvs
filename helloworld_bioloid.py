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
basePosition = [0, 0, 0.215]

baseOrientation = p.getQuaternionFromEuler([0,0,-0.5831853071795866])

bioId = p.loadURDF("../ros-bioloid/src/bioloid_master/urdf/mioloid_robot_head.urdf",basePosition,baseOrientation ,useFixedBase = True )    
print("Bioloid id:"+ str(bioId))

#add debug slider  0.018959998305018582, -0.0030300449845015653, 0.15345262248306862,
init_pos = [  1.1880585171461655 , -2.31561388612597, 1.1892599041779879 , 1.1880174349839125 , -2.3143761387021726  , 1.1886576104218778 ] 
targetPos  = [ 0, 0, 0.15345262248306862,0.0, 0.0, -0.5831853071795866,  1.1880585171461655 , -2.31561388612597, 1.1892599041779879 , 1.1880174349839125 , -2.3143761387021726  , 1.1886576104218778 ] 
norm_joints = []
jointIds=[]
paramIds=[]
jointName=[]
freeJointList = [ 14, 15, 16, 21, 22, 23]
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
for i in range(len(freeJointList)):
        p.setJointMotorControl2(bioId,freeJointList[i], p.POSITION_CONTROL,targetPosition=p.readUserDebugParameter(i),targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=2 )
while True: 
    pos = []
    orn = []
    obs = []
    jointStates = []
    pos, orn = p.getBasePositionAndOrientation(bioId)
    obs.extend(list(pos))
    euler = p.getEulerFromQuaternion(orn)
    obs.extend(list(euler))
    print(obs[3:6])
    for i in range(len(freeJointList)):
        jointStates.append(p.getJointState(bioId, freeJointList[i]))
    jointPoses = [x[0] for x in jointStates]
    obs.extend(list(jointPoses))
   
    i=0
    for i in range(len(freeJointList)):
        p.setJointMotorControl2(bioId,freeJointList[i], p.POSITION_CONTROL,targetPosition=p.readUserDebugParameter(i),targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=25)
    p.stepSimulation()
    d_pos = goal_distance(np.array(obs[0:3]), np.array(targetPos[0:3]))
    d_orn = goal_distance(np.array(obs[3:6]), np.array(targetPos[3:6]))
    sm = 0 
    for i in range(6, 12):
       
        if targetPos[i]> obs[i]:
            sm = sm + np.fabs(targetPos[i] - obs[i])
        else: 
            sm = sm + np.fabs(obs[i] - targetPos[i])
    

    print("d_pos:" + str(d_pos))
    print("d_orn:" + str(d_orn))
    print("sm:" + str(sm))

    time.sleep(0.01)


   		

bioloidPos, bioloidOrn = p.getBasePositionAndOrientation(bioId)
print(bioloidPos,bioloidOrn)
p.disconnect()
