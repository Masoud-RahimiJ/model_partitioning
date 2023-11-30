import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


labels = ["Container and Runtime", "Inference Framework", "Model"]

data = {
    "Torch\nVGG19": [11,180,635],
    "Torch\nVit_L_16": [11,180,1250],
    "Torch\nRegnet_y_128_gf": [11,180,2620],
    "TF\nVGG19": [11,311,680],
    "TF\nConvnext_large": [11,311,1230],
    "TF\nconvnext_xlarge": [11,311,2840],
    "MXnet\nVGG19": [11,26,1200],
}

width=0.2
x_axis = [0]
figure_points = []
ticks=data.keys()

for i in range(3):
    figure_points.append([])
    for k,v in data.items():
        sum_time =  sum(data[k])
        figure_points[-1].append((data[k][i]/sum_time)*100)
    figure_points[-1] = np.array(figure_points[-1])


for i in range(len(data)):
    if i%3==0:
        x_axis.append(x_axis[-1]+0.35)
    else:
        x_axis.append(x_axis[-1]+0.21)
del x_axis[0]     

bottoms=[np.array([0]*len(figure_points[0]))]
for i in range(2):
    bottoms.append(bottoms[-1]+figure_points[i])

plt.figure(figsize=(11,6))
plt.bar(x_axis, figure_points[0], width, color='darkviolet')
plt.bar(x_axis, figure_points[1], width, bottom=bottoms[1], color='cyan')
plt.bar(x_axis, figure_points[2], width, bottom=bottoms[2], color='green')
plt.xticks(x_axis, ticks)
plt.xlabel("Models",  fontdict={"fontsize":16})
plt.ylabel("Percent",  fontdict={"fontsize":16})
plt.legend(labels)
plt.title("Memory Usage", fontdict={"fontsize":20})
plt.show()
