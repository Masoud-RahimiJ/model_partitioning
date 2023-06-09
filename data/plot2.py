import matplotlib.pyplot as plt
import numpy as np
from time import sleep

labels = ["Create Container", "Import Libraries", "Initialize Model", "Download Parameters", "Load Parameters", "Preprocess Input", "Execute", "Others"]
data = {
    "mxnet-resnet152": [0, 0, 0, 0, 0, 0, 0, 0],
    "pytorch-resnet152": [0, 0, 0, 0, 0, 0, 0, 0],
    "mxnet-vgg19": [0, 0, 0, 0, 0, 0, 0, 0],
    "tensorflow-vgg19": [0, 0, 0, 0, 0, 0, 0, 0],
    "pytorch-regnet_y_128gf": [0, 0, 0, 0, 0, 0, 0, 0],
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