import numpy as np
from matplotlib import pyplot as plt
import pickle as pk
import os


def plot(functions, precision=200, low_range=-3, high_range=5):
    x = np.random.uniform(low_range, high_range, precision)
    x = np.sort(x)
    values_x = np.zeros((len(x), len(functions)))
    for i in range(len(x)):
        for j in range(len(functions)):
            values_x[i, j] = functions[j](x[i])
    plt.plot(values_x[:, 0], values_x[:, 1])

def scatter(data, functions):
    values_x = np.zeros((len(data), len(functions)))
    for i in range(len(data)):
        for j in range(len(functions)):
            values_x[i, j] = functions[j](data[i])
    plt.scatter(values_x[:, 0], values_x[:, 1], s=10, color="red")


def save_obj(objects, names, saving_dir):
    assert len(names) == len(objects)
    for name, object in zip(objects):
        pk.dump(object, saving_dir + name + ".pickle")

def load_obj(saving_dir):
    files = os.listdir(saving_dir)
    objects = []
    for file in files:
        temp = open(saving_dir + file)
        objects.append(pk.load(temp))
    return objects
