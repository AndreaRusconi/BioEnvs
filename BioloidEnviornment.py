import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import pybullet as p
import robot_data
import math as m

class iCubEnv:

    def __init__(self, urdfRootPath=robot_data.getDataPath(),
                    timeStep=0.01,
                    useInverseKinematics=0, arm='l', useOrientation=0):

        self.urdfRootPath = os.path.join(urdfRootPath, "iCub/icub_fixed_model.sdf")
        self.timeStep = timeStep
        self.useInverseKinematics = useInverseKinematics
        self.useOrientation = useOrientation
        self.useSimulation = 1

        self.indices_torso = range(12,15)
        self.indices_left_arm = range(15,22)
        self.indices_right_arm = range(25,32)
        self.indices_head = range(22,25)

        self.home_pos_torso = [0.0, 0.0, 0.0] #degrees
        self.home_pos_head = [0.47, 0, 0]

        self.home_left_arm = [-29.4, 40.0, 0, 70, 0, 0, 0]
        self.home_right_arm = [-29.4, 40.0, 0, 70, 0, 0, 0]

        self.workspace_lim = [[0.25,0.52],[-0.2,0.2],[0.5,1.0]]

        self.control_arm = arm if arm =='r' or arm =='l' else 'l' #left arm by default

        self.reset()

    def reset(self):
        self.icubId = p.loadSDF(self.urdfRootPath)[0]
        self.numJoints = p.getNumJoints(self.icubId)