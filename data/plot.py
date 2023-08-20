import matplotlib.pyplot as plt
import numpy as np
from time import sleep

figures = ["Vgg19.csv", "Regnet-y-128gf.csv", "Convnext-xlarge.csv"]
v_lines = [(2.744,2.82), (6.43,7.35), (6.03,6.45)]

def import_data_from_file(file):
    content = file.read().split('\n')
    content = list(map(lambda line: line.split(';'), content))
    x = np.array(list(map(lambda point: (int(point[0]) - int(content[0][0]))/10000, content)))
    y = np.array(list(map(lambda point: int(point[1]), content)))
    return x, y


# for idx, figure in enumerate(figures):
idx=2
figure = figures[idx]
with open(figure) as f:
    plt.figure(figsize=(8,4))
    x, y = import_data_from_file(f)
    plt.subplot(1, 1, 1)
    plt.plot(x,y, color='black')
    plt.xticks([])
    # plt.title(figure.split('.')[0])
    plt.ylabel("Memory Usage(MB)", fontdict={"fontsize":13})
    for v_line in v_lines[idx]:
        plt.axvline(x = v_line, color = 'red')

plt.title("Tensorflow-Convnext-xlarge", fontdict={"fontsize":15})
plt.show()