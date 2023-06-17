import matplotlib.pyplot as plt
import numpy as np
from time import sleep

figures = ["mxnet vgg19.csv", "pytorch regnet-y-128gf.csv", "tensorflow convnext-xlarge.csv"]


def import_data_from_file(file):
    content = file.read().split('\n')
    content = list(map(lambda line: line.split(';'), content))
    x = np.array(range(len(content)))
    y = np.array(list(map(lambda point: int(int(point[1])), content)))
    return x, y


for idx, figure in enumerate(figures):
    with open(figure) as f:
        x, y = import_data_from_file(f)
        plt.subplot(2, 2, idx+1)
        plt.plot(x,y)
        # plt.xticks([])
        plt.title(figure.split('.')[0])
        plt.xlabel("Time(S)")
        plt.ylabel("Memory Usage(MB)")

plt.suptitle("MEMORY USAGE IN CONTAINER LIFE TIME")
plt.show()