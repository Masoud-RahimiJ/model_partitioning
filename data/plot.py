import matplotlib.pyplot as plt
import numpy as np
from time import sleep

figures = ["ll.csv"]
# v_lines = [(2.745,2.81), (15.43,16.32), (8.02,8.45)]

def import_data_from_file(file):
    content = file.read().split('\n')
    content = list(map(lambda line: line.split(' '), content))
    x = np.array(list(map(lambda point: (int(point[0]) - int(content[0][0]))/10000, content)))
    y = np.array(list(map(lambda point: int(point[1]), content)))
    return x, y


for idx, figure in enumerate(figures):
    with open(figure) as f:
        x, y = import_data_from_file(f)
        plt.subplot(1, 1, idx+1)
        plt.plot(x,y)
        plt.xticks([])
        plt.title(figure.split('.')[0])
        plt.xlabel("Time")
        plt.ylabel("Memory Usage(MB)")
        # for v_line in v_lines[idx]:
        #     plt.axvline(x = v_line, color = 'black')

plt.suptitle("MEMORY USAGE WITH MXNET LIB")
plt.show()