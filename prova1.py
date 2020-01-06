from maraboupy import Marabou
import numpy as np

filename = 'models/pb/my_model.pb'#'Marabou/maraboupy/examples/networks/frozen_lisa_model.pb'
network = Marabou.read_tf(filename)

inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0]
