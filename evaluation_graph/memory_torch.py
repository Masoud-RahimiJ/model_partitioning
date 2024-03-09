import matplotlib.pyplot as plt
import numpy as np

# method = [780, 1899, 5688, 1704, 5773, 13243, 1133, 6563, 12647]
method = [653, 1810, 3838, 1704, 5773, 13243, 1133, 6563, 12647]
original = [806, 1936, 7677, 1733, 8262, 18193, 1390, 9067, 17980]


x1=[0,0.8,1.6]
x2=[0.3,1.1,1.9]
x3=[0.15,0.95,1.75]
labels = ["Resnet101", "Vgg19", "Regnet-y-128-gf", "Gpt2-base", "Labse", "Gpt2-xl", "Wav2vec2-base", "Whisper-medium", "Whisper-large-v2"]
width=0.3
tt = ["Small", "Medium", "Large"]
start = 2

plt.figure(figsize=(19,8))
for start in range(3):
    plt.subplot(1,3,start+1)
    plt.bar(x1, method[start::3], width, color='darkviolet')
    plt.bar(x2, original[start::3], width, color='blue')
    plt.xticks(x3, labels[start::3])
    plt.xlabel("Models")
    plt.ylabel("Memory")
    plt.title(tt[start])
    
plt.suptitle("Peak Memory Usage")
plt.show()
