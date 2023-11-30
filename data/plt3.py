import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



labels = ["Container and Runtime", "Inference Framework", "Model"]
#vision
#smal
Vgg19 = {
    "mxnet":  [2.086, 0.005, 0.627, 0.189, 0.262],
    "tensorflow":  [2.732, 0.483, 0.693, 0.45, 0.508],
    "pyotch": [1.777, 1.745, 0.65, 0.2, 0.065],
    "pyotch-gpu": [1.88, 1.87, 0.65, 1.35, 0.619]
}


#text
#small
Gpt2 = {
    "tensorflow":  [3.02, 1.12, 0.663, 0.36, 4],
    "pyotch": [3.51, 3.27, 0.66, 0.18, 0.85]
}
#medium
Gpt2_medium = {
    "tensorflow":  [3, 1.34, 2, 0.83, 8.01],
    "pyotch": [3.49, 6.45, 2.18, 0.58, 2.2]
}
#large
Gpt2_xl = {
    "tensorflow":  [3.232, 1.3, 9.362, 4.26, 19.8],
    "pyotch": [3.577, 22.66, 9.22, 2.1, 6.74],
    # "pyotch-gpu": [3.577, 22.66, 9.22, 9.08, 0.61],
    # "tensorflow-gpu":  [3.232, 1.3, 9.362, 4.26, 19.8],
}

#audio
#small
wav2vec = {
    "tensorflow":  [3.24, 1.11, 0.55, 0.24, 0.7],
    "pyotch": [3.65, 2.23, 0.56, 0.167, 0.52]
}
#medium
Whisper_medium = {
    "tensorflow":  [3.16, 1.37, 4.65, 1.54, 15.04],
    "pyotch": [3.57, 9.6, 4.5, 1, 9.05]
}
#large
Whisper_large = {
    "tensorflow":  [3.32, 1.54, 9.04, 5.2, 28.54],
    "pyotch": [3.57, 17.77, 8.88, 2.12, 16.87]
}

Resnet50_pytorch_cpu= [2,0.47,0.14,0.1,0.05]
####gpu
Resnet50_pytorch = [2,0.47,0.14,1.4,0.61]
Vgg19_pytoch = [2,2.76,0.64,1.78,0.58]
regent_pytorch= [2,12.2,4.2,3.78,0.62]

Gpt2_pytorch= [2.6,3.98,0.66,3.06,0.54]
Gpt2_md_pytorch= [2.6,9.50,2.16,3.96,0.73]
Gpt2_xl_pytorch= [2.6,37.1,9.12,8.2,1.17]

wav_pytorch= [2.6,2.35,0.56,2.05,0.64]
Whisper_pytorch= [2.6,16.13,4.52,4.56,2.47]
Whisper_large_pytorch= [2.6,30.7,8.93,7.22,3.09]


Resnet50_tf = [6.7,1.2,0.11,0.4,2.68]
Vgg19_tf = [6.7,0.3,0.62,1.1,1.7]
regent_tf= [7.2,0.64,4.5,2.07,5.2]

Gpt2_tf= [7.1,0.56,0.63,0.9,4.6]
Gpt2_md_tf= [7.1,0.88,2.11,2.59,8.89]
Gpt2_xl_tf= [7.1,0.84,9.23,10.9,17.3]

wav_tf= [7.1,5.1,0.56,0.58,0.28]
Whisper_medium_tf= [7.1,5.6,4.61,3.96,11.5]
Whisper_large_tf= [7.1,5.8,9.01,9.22,15.61]


data_pytorch_cpu = {
    "Resnet50" : [2,0.01,0.14,0.1,0.05],
    "Vgg19" : [1.877, 0.004, 0.65, 0.2, 0.065],
    "Regent\ny_128gf": [2,0.4,4.12,0.95,0.67],

    "Gpt2": [3.51, 1.27, 0.66, 0.18, 0.85],
    "Gpt2_medium": [3.49, 1.45, 2.18, 0.58, 2.2],
    "Gpt2_xl": [3.577, 1.66, 9.22, 2.1, 6.74],

    "Wav2vec\nbase": [3.65, 2.23, 0.56, 0.167, 0.52],
    "Whisper\nmedium": [3.57, 1.02, 4.5, 1, 9.05],
    "Whisper\nlarge_v2": [3.57, 1.17, 8.88, 2.12, 16.87]
}

data_pytorch_gpu = {
    "Resnet50" : [2,0.02,0.14,1.4,0.61],
    "Vgg19" : [2,0.01,0.64,1.78,0.58],
    "Regent\ny_128gf": [2,0.6,4.2,3.78,0.62],

    "Gpt2": [2.6,1.98,0.66,3.06,0.54],
    "Gpt2_medium": [2.6,2.13,2.16,3.96,0.73],
    "Gpt2_xl": [2.6,2.21,9.12,8.2,1.17],

    "Wav2vec\nbase": [2.6,2.35,0.56,2.05,0.64],
    "Whisper\nmedium": [2.6,1.83,4.52,4.56,2.47],
    "Whisper\nlarge_v2": [2.6,2.14,8.93,7.22,3.09]
}

data_tf_cpu = {
    "Resnet50" : [2.6,0.32,0.14,0.21,0.41],
    "Vgg19" : [2.732, 0.483, 0.693, 0.45, 0.508],
    "Regent\ny_040": [3.2,0.92,4.45,1.9,5.23],

    "Gpt2": [3.02, 1.12, 0.663, 0.36, 4],
    "Gpt2_medium": [3, 1.34, 2, 0.83, 8.01],
    "Gpt2_xl": [3.232, 1.3, 9.362, 4.26, 19.8],

    "Wav2vec\nbase": [3.24, 1.11, 0.55, 0.24, 0.7],
    "Whisper\nmedium": [3.16, 1.37, 4.65, 1.54, 15.04],
    "Whisper\nlarge_v2": [3.32, 1.54, 9.04, 5.2, 28.54]
}

data_tf_gpu = {
    "Resnet50" : [3.7,1.2,0.11,0.4,2.68],
    "Vgg19" : [3.7,0.3,0.62,1.1,1.7],
    "Regent\ny_040": [4.2,0.64,4.5,2.07,5.2],

    "Gpt2": [4.1,0.56,0.63,0.9,4.6],
    "Gpt2_medium": [4.1,0.88,2.11,2.59,8.89],
    "Gpt2_xl": [4.1,0.84,9.23,10.9,17.3],

    "Wav2vec\nbase": [4.1,2.1,0.56,0.58,0.28],
    "Whisper\nmedium": [4.1,2.6,4.61,3.96,11.5],
    "Whisper\nlarge_v2": [4.1,2.8,9.01,9.22,15.61]
}

data_ll = {
    "Torch\nVGG19": [11,180,1000],
    "Torch\nVit_L_16": [11,180,2300],
    "Torch\nRegnet_y_128_gf": [11,180,4800],
    "TF\nVGG19": [11,311,1800],
    "TF\nConvnext_large": [11,311,2300],
    "TF\nconvnext_xlarge": [11,311,3400],
    "MXnet\nVGG19": [11,26,1200],
}

# data={
#     "image": {
#         "small": {"Vgg19": Vgg19}
#     },
#     "text": {
#         "small": {"Gpt2": Gpt2},
#         "medium": {"Gpt2_md": Gpt2_medium},
#         "large": {"Gpt2_xl": Gpt2_xl},
#     },
#     "voice": {
#         "small":{"wav2vec": wav2vec} ,
#         "medium": {"Whisper\nmd": Whisper_medium},
#         "large": {"Whisper\nl": Whisper_large},
#     }
# }

data=data_ll
# data=data_tf_cpu
# data=data_pytorch_gpu
# data=data_pytorch_cpu
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

width=0.2
plt.figure(figsize=(11,6))
plt.bar(x_axis, figure_points[0], width, color='darkviolet')
plt.bar(x_axis, figure_points[1], width, bottom=bottoms[1], color='cyan')
plt.bar(x_axis, figure_points[2], width, bottom=bottoms[2], color='green')
# plt.bar(x_axis, figure_points[3], width, bottom=bottoms[3], color='red')
# plt.bar(x_axis, figure_points[4], width, bottom=bottoms[4], color='orange')
plt.xticks(x_axis, ticks)
plt.xlabel("Models",  fontdict={"fontsize":16})
plt.ylabel("percent",  fontdict={"fontsize":16})
plt.legend(labels)
plt.title("Memory Usage", fontdict={"fontsize":20})



plt.show()
