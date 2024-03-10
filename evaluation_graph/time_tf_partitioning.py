import matplotlib.pyplot as plt 
import numpy as np

y = [0.18, 0.45, 2.23, 0.42, 3.12, 5.26]
x = ["Resnet101", "Vgg19", "Regnet-y-128-gf", "Gpt2-base", "Labse", "Gpt2-xl"]#,"Wav2vec2-base", "Whisper-medium", "Whisper-large-v2"]


plt.bar(np.array(x), np.array(y))
plt.xlabel('Models')
plt.ylabel('Second')
plt.title("TensorFlow Partitioning Time")
plt.show()