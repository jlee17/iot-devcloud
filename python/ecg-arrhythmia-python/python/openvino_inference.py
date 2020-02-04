import os
from time import time

import numpy as np
import scipy.stats as sst
import sklearn.metrics as skm
from openvino.inference_engine import IENetwork, IECore
from tqdm import tqdm

import load

model_path = "./0.427-0.863-020-0.290-0.899.hdf5"
data_csv = "./data/reference.csv"

print("Loading Dataset")
ecgs, labels = load.load_dataset(data_csv)

# Load network and add CPU extension
ie = IECore()
ie.add_extension('/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so',"CPU")
net = IENetwork(model = './models/output_graph.xml', weights = './models/output_graph.bin')
exec_net = ie.load_network(network=net, device_name='CPU')

print("Starting Inference")
probs_total = []
total_time = 0
for x in tqdm(ecgs):
    x = load.process_x(x)
    start_time = time()
    res = exec_net.infer(inputs={"inputs": x})
    total_time += (time() - start_time)
    probs = res["time_distributed_1/Reshape_1/Softmax"]
    probs_total.append(probs)

print("OpenVINO took {} sec for inference".format(total_time))

# The class distribution of the overall dataset
prior = [[[0.15448743, 0.66301941, 0.34596848, 0.09691286]]]

# Determine the predicted class from the most commonly predicted class 
preds = []
for p in probs_total:
    preds.append(sst.mode(np.argmax(p / prior, axis=2).squeeze())[0][0])
    
# Generate a report with the precision, recall, and f-1 scores for each of the classes
report = skm.classification_report(labels, preds, target_names=['A','N','O','~'], digits=3)
scores = skm.precision_recall_fscore_support(labels, preds, average=None)

print(report)
print ("CINC Average {:3f}".format(np.mean(scores[2][:3])))