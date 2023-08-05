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
vgg19 = {
    "mxnet":  [2.086, 0.005, 0.627, 0.189, 0.262],
    "tensorflow":  [2.732, 0.483, 0.693, 0.45, 0.508],
    "pyotch": [1.777, 1.745, 0.65, 0.2, 0.065],
    "pyotch-gpu": [1.88, 1.87, 0.65, 1.35, 0.619]
}


#text
#small
gpt = {
    "tensorflow":  [3.02, 1.12, 0.663, 0.36, 4],
    "pyotch": [3.51, 3.27, 0.66, 0.18, 0.85]
}
#medium
gpt_medium = {
    "tensorflow":  [3, 1.34, 2, 0.83, 8.01],
    "pyotch": [3.49, 6.45, 2.18, 0.58, 2.2]
}
#large
gpt_xl = {
    "tensorflow":  [3.232, 1.3, 9.362, 4.26, 19.8],
    "pyotch": [3.577, 22.66, 9.22, 2.1, 6.74],
    "pyotch-gpu": [3.577, 22.66, 9.22, 9.08, 0.61],
    "tensorflow-gpu":  [3.232, 1.3, 9.362, 4.26, 19.8],
}

#audio
#small
wav2vec = {
    "tensorflow":  [3.24, 1.11, 0.55, 0.24, 0.7],
    "pyotch": [3.65, 2.23, 0.56, 0.167, 0.52]
}
#medium
whisper_medium = {
    "tensorflow":  [3.16, 1.37, 4.65, 1.54, 12.04],
    "pyotch": [3.57, 9.6, 4.5, 1, 9.05]
}
#large
whisper_large = {
    "tensorflow":  [3.32, 1.54, 9.04, 5.2, 22.54],
    "pyotch": [3.57, 17.77, 8.88, 2.12, 16.87]
}

resnet50_pytorch_cpu= [2,0.47,0.14,0.1,0.05]
####gpu
resnet50_pytorch = [2,0.47,0.14,1.4,0.61]
vgg19_pytoch = [2,2.76,0.64,1.78,0.58]
regent_pytorch= [2,12.2,4.2,3.78,0.62]

gpt_pytorch= [4.6,3.98,0.66,3.06,0.54]
gpt_md_pytorch= [4.6,9.50,2.16,3.96,0.73]
gpt_xl_pytorch= [4.6,37.1,9.12,8.2,1.17]

wav_pytorch= [4.6,2.35,0.56,2.05,0.64]
whisper_pytorch= [4.6,16.13,4.52,4.56,2.47]
whisper_large_pytorch= [4.6,30.7,8.93,7.22,3.09]


resnet50_pytorch = [2,0.47,0.14,1.4,0.61]
vgg19_pytoch = [2,2.76,0.64,1.78,0.58]
regent_pytorch= [2,12.2,4.2,3.78,0.62]




resnet50_tf = [6.7,1.2,0.11,0.4,2.68]
vgg19_tf = [6.7,0.3,0.62,1.1,1.7]
regent_tf= [7.2,0.64,4.5,2.07,5.2]

gpt_tf= [7.1,0.56,0.63,0.9,4.6]
gpt_md_tf= [7.1,0.88,2.11,2.59,8.89]
gpt_xl_tf= [7.1,0.84,9.23,10.9,17.3]

data={
    "image": {
        "small": {"vgg19": vgg19}
    },
    "text": {
        "small": {"gpt": gpt},
        "medium": {"gpt_md": gpt_medium},
        "large": {"gpt_xl": gpt_xl},
    },
    "voice": {
        "small":{"wav2vec": wav2vec} ,
        "medium": {"whisper_md": whisper_medium},
        "large": {"whisper_l": whisper_large},
    }
}

x_axis = [0]
figure_points = []
ticks=[]
for i in range(5):
    figure_points.append([])
    for category in data:
        add_to_x = 0.5
        for size in data[category]:
            for name in data[category][size]:
                for library in data[category][size][name]:
                    sum_time =  sum(data[category][size][name][library])
                    figure_points[-1].append(data[category][size][name][library][i]/sum_time)
                    if i == 0:
                        x_axis.append(x_axis[-1]+add_to_x)
                        ticks.append(f"{category}\n{size}\n{library}\n{name}")
                    add_to_x = 0.21
            add_to_x = 0.3
    figure_points[-1] = np.array(figure_points[-1])

del x_axis[0]
bottoms=[np.array([0]*len(figure_points[0]))]
for i in range(4):
    bottoms.append(bottoms[-1]+figure_points[i])

width=0.2

plt.bar(x_axis, figure_points[0], width, color='darkviolet')
plt.bar(x_axis, figure_points[1], width, bottom=bottoms[1], color='cyan')
plt.bar(x_axis, figure_points[2], width, bottom=bottoms[2], color='green')
plt.bar(x_axis, figure_points[3], width, bottom=bottoms[3], color='red')
plt.bar(x_axis, figure_points[4], width, bottom=bottoms[4], color='orange')
plt.xticks(x_axis, ticks)
plt.xlabel("Models",  fontdict={"fontsize":15})
plt.ylabel("percent",  fontdict={"fontsize":15})
plt.legend(labels)
plt.title("tasks time percentage in cold start", fontdict={"fontsize":20})



plt.show()
