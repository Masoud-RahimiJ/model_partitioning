import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# data = {
#     "mxnet-resnet152": [0.502, 1.404, 0.045, 0.449, 0.196, 0.288, 0.109],
#     "pytorch-resnet152": [0.534, 1.162, 0.759, 0.324, 0.118, 0.121, 0.117],
#     "mxnet-vgg19": [0.538, 1.437, 0.005, 0.957, 0.189, 0.262, 0.110],
#     "tensorflow-vgg19": [0.531, 2.091, 0.483, 0.943, 0.45, 0.508, 0.110],
#     "pytorch-regnet_y_128gf": [0.552, 1.149, 7.831, 3.975, 0.954, 0.673, 0.115],
#     "tensorflow-convnext_xlarg": [0.54, 2.067, 2.846, 2.405, 1.03, 6.32, 0.110],
#     "transformer-gpt2_big": [0.565, 1.375, 13.96, 4.795, 1.12, 4.033, 0.113],
#     ""pyotch-vgg19": []"
# }

labels = ["Container and Library Initialization", "Initialize Model", "Download Model", "Load Model",  "Execute"]
#vision
#smal
vgg19_data = {
    "mxnet":  [2.086, 0.005, 0.627, 0.189, 0.262],
    "tensorflow":  [2.732, 0.483, 0.693, 0.45, 0.508],
    "pyotch": [1.777, 1.745, 0.65, 0.2, 0.065]
}


#text
#small
gpt_data = {
    "tensorflow":  [3.02, 1.12, 0.663, 0.36, 4],
    "pyotch": [3.51, 3.27, 0.66, 0.18, 0.85]
}
#medium
gpt_medium_data = {
    "tensorflow":  [3, 1.34, 2, 0.83, 8.01],
    "pyotch": [3.49, 6.45, 2.18, 0.58, 2.2]
}
#large
gpt_xl_data = {
    "tensorflow":  [3.232, 1.3, 9.362, 4.26, 19.8],
    "pyotch": [3.577, 22.66, 9.22, 2.1, 6.74]
}

#audio
#small
wav2vec_data = {
    "tensorflow":  [3.24, 1.11, 0.55, 0.24, 0.7],
    "pyotch": [3.65, 2.23, 0.56, 0.167, 0.52]
}
#medium
whisper_medium_data = {
    "tensorflow":  [3.16, 1.37, 4.65, 1.54, 0.7],
    "pyotch": [3.65, 2.23, 0.56, 0.167, 0.52]
}
#large
whisper_large_data = {
    "tensorflow":  [3.32, 1.11, 0.55, 0.24, 0.7],
    "pyotch": [3.65, 2.23, 0.56, 0.167, 0.52]
}


sum_time = {k: sum(data[k]) for k in data}

figure_points = []
for i in range(5):
    figure_points.append([])
    for k in data:
        figure_points[-1].append(data[k][i]/sum_time[k])
    figure_points[-1] = np.array(figure_points[-1])


fig, ax1 = plt.subplots()

ax1.bar(list(data.keys()), figure_points[0], color='darkviolet')
ax1.bar(list(data.keys()), figure_points[1], bottom=figure_points[0], color='cyan')
ax1.bar(list(data.keys()), figure_points[2], bottom=figure_points[0]+figure_points[1], color='green')
ax1.bar(list(data.keys()), figure_points[3], bottom=figure_points[0]+figure_points[1]+figure_points[2], color='red')
ax1.bar(list(data.keys()), figure_points[4], bottom=figure_points[0]+figure_points[1]+figure_points[2]+figure_points[3], color='orange')

# ax1.bar(list(data.keys()), [0,0,0,0,0,0,0], color='black')
ax1.set_xlabel("Library",  fontdict={"fontsize":15})
ax1.set_ylabel("percent",  fontdict={"fontsize":15})
ax1.legend(labels)
ax1.set_title("VGG19-vision-small", fontdict={"fontsize":20})

ax2 = ax1.twinx()

ax2.plot(data.keys(), sum_time.values(), label="Sum Time", color="black", marker = '.')
ax2.tick_params(axis='y', labelcolor="black")

fig.tight_layout()
plt.show()
