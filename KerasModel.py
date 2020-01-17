from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from BioloidEnviornmentHER_fixed import bioEnv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
import numpy 
import time
from gym import spaces

#salvataggio modello in formato .pb per Marabou 
def savePb(model):
	frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
	tf.train.write_graph(frozen_graph, "Maraboy_model", "my_model.pb", as_text=False)

#freeze session per salvataggio .pb per Marabou 
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
#trasformazione observation da dict (Stable_baseline) a numpy
def dictToArray(dict):
    obs = []
    o = []
    o.extend(dict['observation'])
    o.extend(dict['achieved_goal'])
    o.extend(dict['desired_goal'])
    obs.append(o)
    obs= numpy.asarray(obs)
    return obs
#squat test 
def Squat(obs):

    for i in range(1000):
        action= model_K.predict(obs)
        print(action)
        obs, reward, done , _= env.step(action[0])
        obs = dictToArray(obs)
        
        if done:
            print(i)
            obs = env.reset()
            print(obs)
            obs = dictToArray(obs)
#____________________________________________________________


batch_size = 256
num_classes = 1
n_input = 24
output = [] 
init_obs = numpy.asarray([[0.,0.,0.215 ,0.,0.,-0.5831853071795866  ,0.,0.,0.,0.,0.,0.,    0.,0.,0.215,0.,0.,-0.5831853071795866      ,0.,0.,0.16, 0.0,0.0,-0.5831853071795866]])
first_state = [0,0,0.215 ,0,0,-0.5831853071795866  ,0,0,0,0,0,0,    0,0,0.215,0,0,-0.5831853071795866]
target =  numpy.asarray([ 0, 0, 0.16,     0.0, 0.0, -0.5831853071795866] )

#loading weights
fc0_weights = numpy.load ("modello/parameterss/model/pi/fc0/kernel:0.npy")
fc0_biases = numpy.load ("modello/parameterss/model/pi/fc0/bias:0.npy")
fc0  = [fc0_weights, fc0_biases]
fc1_weights = numpy.load ("modello/parameterss/model/pi/fc1/kernel:0.npy")
fc1_biases = numpy.load ("modello/parameterss/model/pi/fc1/bias:0.npy")
fc1 = [fc1_weights, fc1_biases]
dense_weights = numpy.load ( "modello/parameterss/model/pi/dense/kernel:0.npy")
dense_biases = numpy.load ( "modello/parameterss/model/pi/dense/bias:0.npy")
dense = [dense_weights, dense_biases]

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

model_K.summary()

#S_BASELINES
env = bioEnv()
obs_dict = env.reset()
obs = dictToArray(obs_dict)
Squat(obs)
