import numpy as np
import os
import random
import pandas as pd
from enum import Enum
from sklearn import preprocessing
import itertools
import math
import targets
import argparse

class Feature(Enum):
    timestamp = 0
    length = 1
    direction = 2
    wang = 3

TS = "timestamp"
DIR = "direction"
LEN = "length"
NUM = "num_cells"
ACK = "ack"
WANG = "wang"


def expand2(dir, num_cells):
    expanded = []
    for _ in range(0, num_cells):
        expanded.append(dir)
    return expanded

def expand(num_cells):
    dir = int(math.copysign(1, num_cells))
    num_cells = abs(num_cells)
    expanded = []
    for _ in range(0, num_cells):
        expanded.append(dir)
    return expanded

def to_wang(trace, maxlen):
    # truncate to maxlen
    trace = trace[:maxlen]
    # pad to maxlen
    trace = np.pad(trace, (0, maxlen - trace.size), 'constant')

    trace.astype(np.int8)
    return trace

def parse_target(filename):
    return filename.split("-")[0]

def smart_shuffle(lists, list=False):
    if list:
        return [x for x in itertools.chain.from_iterable(itertools.zip_longest(*lists)) if x is not None and x.any()]
    else:
        return [x for x in itertools.chain.from_iterable(itertools.zip_longest(*lists)) if x]

def save_data(txtname, savepath, maxlen, traces, towang=True):
    """Saves the dataset.
    # Arguments
        txtname: path to txt file with list of files.txt with traffic traces
        savepath: path to file where to save the dataset
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    fnames = [x.strip() for x in open(txtname, "r").readlines()]
    #random.shuffle(fnames)

    nb_traces = len(fnames)

    print("Saving dataset with {} traces".format(nb_traces))

    labels = np.empty([nb_traces], dtype=np.dtype(object))
    classes = np.empty([nb_traces], dtype=np.dtype(object))
    data = np.empty([nb_traces, maxlen], dtype=np.int16)


    num_labels = {}
    numlabels = {}

    j = 0
    for f in fnames:
        label = parse_target(os.path.basename(f))
        num = "0123456789_"
        df = pd.read_csv(f, nrows=1000, sep=";", dtype=None, usecols=[3, 2], header=0)  # 3=direction, 2=length
        # df = pd.read_csv(f, nrows=10000, sep=";", dtype=None, usecols=[1, 2], header=0)  #timestamp=1
        # df = pd.read_csv(f, nrows=10000, sep=";", dtype=None, usecols=[2], header=0)

        if df.empty:
            continue
        for char in num:
            label = label.replace(char, "")
        if label not in num_labels:
            num_labels[label] = 1
        else:
            count = num_labels[label]
            num_labels[label] = count + 1
    contador=0
    no=[]
    yes=[]


    for i in num_labels:
        if num_labels[i]<traces:
            no.append(i)

    for f in fnames:
        label = parse_target(os.path.basename(f))
        num = "0123456789_"
        df = pd.read_csv(f, nrows=1000, sep=";", dtype=None, usecols=[3, 2], header=0)  # 3=direction, 2=length
        # df = pd.read_csv(f, nrows=10000, sep=";", dtype=None, usecols=[1, 2], header=0)  #timestamp=1
        # df = pd.read_csv(f, nrows=10000, sep=";", dtype=None, usecols=[2], header=0)
        if (label ==""):
            continue
        if df.empty:
            continue
        for char in num:
            label = label.replace(char, "")
            clase = label
        if traces:
            if label in no:
                continue

            if label not in numlabels:
                numlabels[label] = 1
                yes.append(label)

            else:
                count = numlabels[label]
                #if count == traces:
                if count== traces:

                    continue
                else:
                    numlabels[label] = count + 1

        if towang:


            values = to_wang((df["direction"] * (df["length"])), maxlen)

        else:
            print("WRONG")
            values = df.values
        classes[j] = clase
        contador=contador+1
        data[j] = values
        labels[j] = label
        j += 1

    labels = np.array(labels)
    nb_classes = len(yes)

    classes = np.array(classes)
    data = np.array(data)
    labels = labels[:contador]
    data=data[:contador]

    classes = classes[:contador]

    # Save in file.npz
    savepath = savepath + '40flowsXapp100length.npz'
    np.savez_compressed(savepath, data=data, labels=labels, classes=classes)
    print('Saved a dataset with {} traces for {} websites to {}'.format(data.shape[0], nb_classes, savepath))

if __name__ == "__main__":
    #URLs = bad_targets()
    #print(URLs)
    parser = argparse.ArgumentParser(description='Save files.txt from txt to file.npz dataset')

    parser.add_argument('--file', '-f',
                        type=str,
                        default="../results/pcapfiles",
                        help='txt file with list of files.txt')
    parser.add_argument('--out', '-o',
                        type=str,
                        default="../file.npz/",
                        help='output dataset name')
    parser.add_argument('--maxlen', '-m',
                        type=int,
                        default=100,
                        help='max amount of features')
    parser.add_argument('--traces', '-t',
                        type=int,
                        default=40,
                        help='amount of traces')

    args = parser.parse_args()
    save_data(args.file, args.out, args.maxlen, args.traces)