import matplotlib.pyplot as plt 


resnet = [4.15, 4.17, 4.2, 4.26, 4.47, 4.93]
resnetNP = [5.19, 5.22, 5.24, 5.38, 5.62, 5.9]

vgg19 = [3.2, 3.21, 3.26, 3.37, 3.53, 3.88]
vgg19NP = [3.3, 3.32, 3.35, 3.48, 3.68, 4]

regnet = [10.4, 10.65, 11.9, 15.15, 20.6, 31.4]
regnetNP = [11.29, 11.61, 12.43, 15.64, 21.47, 33.11]

gpt = [4.8, 4.95, 5.37, 6.1, 7.5, 10.6]
gptNP = [5.1, 5.24, 5.63, 6.33, 7.82, 11]

labse = [12.65, 12.9, 14.4, 17.7, 23.6, 36]
labseNP = [12.66, 12.95, 14.47, 17.8, 23.7, 36.1]

gpt_xl = [17.6, 18.4, 21, 23.4, 28.4, 39]
gpt_xlNP = [18.36, 19, 21.35, 23.6, 28.9, 40]

# wav = [5.05, 5.5, 6.6, 9, 13.2, 21.8]
# wavNP = [5.65, 5.9, 7, 9.4, 13.7, 22.4]

# whisper_md = [22.9, 24.2, 32.5, 44.4, 62.7, 102]
# whisper_mdNP = [25.2, 25.9, 34.4, 45.8, 63.8, 103.2]

# whisper_large = [43.1, 45.5, 58.8, 78.1, 112.8, 173]
# whisper_largeNP = [56.8, 60.3, 67.2, 85.5, 117.8, 178.5]


vision_figures = [resnet, resnetNP, vgg19, vgg19NP, regnet, regnetNP]
vision_label = ["Resnet101", "Vgg19", "Regnet-y-128-gf"]
nltp_figures = [gpt, gptNP, labse, labseNP, gpt_xl, gpt_xlNP]
nltp_label = ["Gpt2-base", "Labse", "Gpt2-xl"]


def draw(labels, figures, idx, title):
    plt.subplot(1,2,idx)
    x_axis = ["1","2","4","8","16","32"]
    y_axis = []
    for i in range(0, len(figures), 2):
        y_axis.append([])
        for j in range(len(figures[0])):
            y_axis[-1].append(100*((figures[i+1][j] / figures[i][j])-1))

    for i in range (3):
        plt.plot(x_axis, y_axis[i], label=labels[i], marker='o', linestyle='dashed') 

    plt.xlabel('Batch Size')
    plt.ylabel('Overhead percent')
    plt.title(title) 
    plt.yticks([0,10,20,30])
    plt.legend() 
  
draw(vision_label, vision_figures, 1, "Vision")
draw(nltp_label, nltp_figures, 2, "NLTP")
# function to show the plot 
plt.suptitle("TensorFlow")
plt.show() 