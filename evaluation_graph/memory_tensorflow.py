import matplotlib.pyplot as plt
import numpy as np


method = [972, 2420, 5140, 1777, 7783, 9390]
original = [981, 2440, 6257, 1779, 8779, 13301]


x1=[0,0.8]
x2=[0.3,1.1]
x3=[0.15,0.95]
labels = ["Resnet101", "Vgg19", "Regnet-y-128-gf", "Gpt2-base", "Labse", "Gpt2-xl"]
width=0.3
tt = ["Small", "Medium", "Large"]

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
