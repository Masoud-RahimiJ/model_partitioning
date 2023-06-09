import matplotlib.pyplot as plt
import numpy as np
from time import sleep

labels = ["Create Container", "Import Libraries", "Initialize Model", "Download Parameters", "Load Parameters", "Preprocess Input", "Execute", "Others"]
data = {
    "mxnet-resnet152": [0.502, 1.404, 0.045, 0.449, 0.196, 0.028, 0.288, 0.109],
    "pytorch-resnet152": [0.534, 1.162, 0.759, 0.324, 0.118, 0.031, 0.121, 0.117],
    "mxnet-vgg19": [0.538, 1.437, 0.005, 0.957, 0.189, 0.111, 0.262, 0.110],
    "tensorflow-vgg19": [0, 0, 0, 0, 0, 0, 0, 0],
    "pytorch-regnet_y_128gf": [0.552, 1.149, 7.831, 3.975, 0.954, 0.031, 0.673, 0.115],
    "tensorflow-convnext_xlarg": [0, 0, 0, 0, 0, 0, 0, 0],
    "transformer-gpt2": [0, 0, 0, 0, 0, 0, 0, 0]
}
cols = ['r','b','c','g', 'orange', 'purple', 'yellow', 'white']

idx = 1
for k, v in data:
    plt.subplot(1, 7, idx)
    plt.pie(v,
    labels =labels,
    colors = cols,
    startangle = 90,
    # shadow = True,
    # explode =(0,0.1,0,0,0),
    autopct ='%1.1f%%')
    plt.title(k)
    idx += 1

plt.suptitle("TASKS TIME PERCENTAGE IN COLD SART")
plt.show()