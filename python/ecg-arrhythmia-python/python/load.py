import csv
import keras
import numpy as np
import os
import random
import scipy.io as sio
import tqdm
import pickle

STEP = 256

from time import time
from pathlib import Path
import sys
sys.path.insert(0, os.path.join(Path.home(), 'Reference-samples/iot-devcloud'))
from demoTools.demoutils import progressUpdate

mean = 7.4661856 
std = 236.10312 
classes = {'A' : 0, 'N' : 1, 'O' : 2, '~' : 3}

def process_x(x):
    x = np.expand_dims(x,axis=0)
    x = (x - mean) / std
    x = x[:, :, None]
    return x

def pad(x, val=0, dtype=np.float32):
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded

def load_dataset(data_csv):
    with open(data_csv, 'r') as fid:
        data = csv.reader(fid)
        rows = [row for row in data]
        names = [row[0] for row in rows]
        ecgs = []
        labels = [classes[row[1]] for row in rows]
        sample_count = len(names)
        time_start = time()
        for i, d in enumerate(tqdm.tqdm(names)):
            ecgs.append(load_ecg('./data/' + d + '.mat'))
            progressUpdate('./logs/' + os.environ['PBS_JOBID']  + '_load.txt', time()-time_start, i+1, sample_count)        
        sizes = []
        for item in ecgs:    
            sizes.append(len(item))
        return ecgs, labels

def load_ecg(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()
    else: # Assumes binary 16 bit integers
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)

    trunc_samp = STEP * int(len(ecg) / STEP)
    return ecg[:trunc_samp]
