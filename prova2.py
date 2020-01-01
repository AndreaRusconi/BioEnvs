from __future__ import print_function

from tensorflow import keras
from BioloidEnviornmentHER_fixed import bioEnv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import numpy 
from gym import spaces

def get_dict(dict):
    obs = []
    o = []
    o.extend(dict['observation'])
    o.extend(dict['achieved_goal'])
    o.extend(dict['desired_goal'])
    obs.append(o)
    obs= numpy.asarray(obs)
    return obs
    
def Squat():
	env = bioEnv()
	obs_dict = env.reset()
	obs = get_dict(obs_dict)
	for i in range(10000):
	    action= model_K.predict(obs)
	    print(action)
	    obs, reward, done , _= env.step(action[0])
	    obs = get_dict(obs)
	    
	    if done:
	        print(i)
	        obs = env.reset()
	        print(obs)
	        obs = get_dict(obs)


batch_size = 256
num_classes = 1
n_input = 24
output = [] 
init_obs = numpy.asarray([[0,0,0.215 ,0,0,-0.5831853071795866  ,0,0,0,0,0,0,    0,0,0.215,0,0,-0.5831853071795866      ,0,0,0.16, 0.0,0.0,-0.5831853071795866]])
first_state = [0,0,0.215 ,0,0,-0.5831853071795866  ,0,0,0,0,0,0,    0,0,0.215,0,0,-0.5831853071795866]
target =  numpy.asarray([ 0, 0, 0.16,     0.0, 0.0, -0.5831853071795866] )

#loading weights
"""
('observation', array([-4.71057630e-04, -3.41648131e-04,  2.15459472e-01,  4.66797351e-04, 
						3.56099666e-03, -5.81670124e-01,  1.42044861e-02, -1.23647752e-02,
       					-1.11741368e-02,  1.26716811e-02, -1.51697683e-02,  1.18522536e-02])), 
('achieved_goal', array([-4.71057630e-04, -3.41648131e-04,  2.15459472e-01,  4.66797351e-04,3.56099666e-03, -5.81670124e-01])),
 ('desired_goal', array([ 0.        ,  0.        ,  0.16      ,  0.        ,  0.        ,-0.58318531]))])

"""
fc0_weights = numpy.load ("modello/parameterss/model/pi/fc0/kernel:0.npy")
fc0_biases = numpy.load ("modello/parameterss/model/pi/fc0/bias:0.npy")
fc0  = [fc0_weights, fc0_biases]

fc1_weights = numpy.load ("modello/parameterss/model/pi/fc1/kernel:0.npy")
fc1_biases = numpy.load ("modello/parameterss/model/pi/fc1/bias:0.npy")
fc1 = [fc1_weights, fc1_biases]

dense_weights = numpy.load ( "modello/parameterss/model/pi/dense/kernel:0.npy")
dense_biases = numpy.load ( "modello/parameterss/model/pi/dense/bias:0.npy")
dense = [dense_weights, dense_biases]
print(dense_weights.shape)

#building model
model_K = Sequential()
model_K.add(Dense(24, input_shape = (24, ), trainable = False, use_bias = False,kernel_initializer = "Identity"))
model_K.add(Dense(64, activation = 'relu', trainable = False))
model_K.add(Dense(64, activation = 'relu', trainable = False))
model_K.add(Dense(6, activation='tanh', trainable = False))

#setting weights

model_K.layers[1].set_weights(fc0)
model_K.layers[2].set_weights(fc1)
model_K.layers[3].set_weights(dense)


#check pesi 


model_K.summary()


