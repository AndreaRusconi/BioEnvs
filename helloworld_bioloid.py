import pybullet as p
import time
import pybullet_data
import math 

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
 
print(pybullet_data.getDataPath())
#Set gravity 
p.setGravity(0,0,-9.8)

#loading Plane
planeId = p.loadURDF("plane.urdf")

#start pos and orientation
bioloidStartPos = [0,0,0.50]
bioloidStartOrientation = p.getQuaternionFromEuler([0,0,30])

#loading model
bioIds = p.loadURDF("ros-bioloid/src/bioloid_master/urdf/mioloid.urdf", bioloidStartPos, bioloidStartOrientation)
print(bioIds)

#add debug slider
jointIds=[]
paramIds=[]
joints_num = p.getNumJoints(bioIds)

#print("len init_pos ",len(init_pos))
print("Number of joints:"+ str(joints_num))
#
#for j in range(joints_num):
#    info = p.getJointInfo(icubId,j)
#    jointName = info[1]
#    jointIds.append(j)
#    paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), info[8], info[9], init_pos[j]/180*m.pi))
#while True:
#    for i in range(joints_num):
#        p.setJointMotorControl2(bioI, i, p.POSITION_CONTROL,
#    								targetPosition=p.readUserDebugParameter(i),
#    								targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)
#
#    	p.stepSimulation()
#    	time.sleep(0.01)
#
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
bioloidPos, bioloidOrn = p.getBasePositionAndOrientation(bioIds)
print(bioloidPos,bioloidOrn)
p.disconnect()
