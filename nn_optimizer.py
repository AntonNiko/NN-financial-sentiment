import nn_model as nn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import itertools
import matplotlib.pyplot as plt
import numpy as np
import csv, json

RESULT_FILE = "1m_sgd.csv"
JSON_FILE = "nn_1m.json"

class Optimizer():
    def __init__(self):
        self.net = nn.Network()
        self.results = []
        

    def test_net(self, variables, record = True):
        """
        Function tests network through the various variables 
        NOTE: Input length, number of layers cannot be changed
        """

        ## Load data for training and prediction
        with open(JSON_FILE) as f:
            data = json.loads(f.read())
            in_data = np.asarray([x[0] for x in data])
            out_data = np.asarray([x[1] for x in data])       

        ## Cycle through every call, and record percentage result for each call
        vars_list = list(variables.values())
        vars_keys = list(variables.keys())
        vars_comb = list(itertools.product(*vars_list))
        print("Number of calls: {}".format(len(vars_comb)))

        for comb in vars_comb:
            ## For each call, set vars in nn_model to current vars
            for i in range(len(comb)):
                nn.variables[vars_keys[i]] = comb[i]
            ## Train and collect prediction for each combination
            self.net.train(in_data, out_data)
            rate = self.net.predict(in_data, out_data)
            ## Setup data to be saved to results csv
            result_list = nn.variables
            result_list["rate"] = rate
            self.results.append(result_list)
            
        ## Save results in file
        fields = vars_keys
        fields.append("rate")
        with open(RESULT_FILE, "w") as f:
            writer = csv.DictWriter(f, delimiter=",", fieldnames=fields)
            writer.writeheader()
            for line in self.results:
                writer.writerow(line)  

if __name__ == "__main__":
    opt = Optimizer()
    opt.test_net({"train_epochs":[5],  
                   "batch_size":[32],
                   "lstm_units":[[100,100]], 
                   "dropout_rate":[[0.2, 0.2]],
                   "activation_final":["softmax"],
                   "loss_function":["binary_crossentropy"],
                   "learning_rate":[0.1]})


