import matplotlib.pyplot as plt 
import numpy as np

y = [0.25, 0.49, 2.15, 0.52, 2.6, 5.27, 0.34, 2.72, 5.25]
x = ["Resnet101", "Vgg19", "Regnet-y-128-gf", "Gpt2-base", "Labse", "Gpt2-xl","Wav2vec2-base", "Whisper-medium", "Whisper-large-v2"]


plt.bar(np.array(x), np.array(y))
plt.xlabel('Models')
plt.ylabel('Second')
plt.title("Pytorch Partitioning Time")
plt.show()