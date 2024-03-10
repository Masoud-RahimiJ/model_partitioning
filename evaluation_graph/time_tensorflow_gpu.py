import matplotlib.pyplot as plt 


resnet = [10.36, 10.37, 10.4, 10.49, 10.7, 11.1]
resnetO = [10.9, 10.93, 10.97, 11.02, 11.13, 11.5]

vgg19 = [8.2, 8.3, 8.6, 9.2, 10.3, 12.8]
vgg19O = [9.3, 9.5, 9.9, 10.6, 11.9, 14.6]

regnet = [25, 30, 37.5, 55, 88, 149]
regnetO = [33, 38, 47, 64, 96, 160]

gpt = [9.2, 9.7, 10.7, 13, 17.5, 26]
gptO = [9.6, 10.2, 11.2, 13.5, 18, 26.5]

labse = [27.7, 29.7, 34.1, 42.5, 58.7, 91.5]
labseO = [30.2, 31.9, 36.4, 44.8, 61.5, 95.2]

gpt_xl = [43, 46, 52, 62.7, 86.6, 133]
gpt_xlO = [47.4, 50.8, 57.1, 68.5, 92, 140]

# wav = [5.05, 5.5, 6.6, 9, 13.2, 21.8]
# wavO = [5.65, 5.9, 7, 9.4, 13.7, 22.4]

# whisper_md = [22.9, 24.2, 32.5, 44.4, 62.7, 102]
# whisper_mdO = [25.2, 25.9, 34.4, 45.8, 63.8, 103.2]

# whisper_large = [43.1, 45.5, 58.8, 78.1, 112.8, 173]
# whisper_largeO = [56.8, 60.3, 67.2, 85.5, 117.8, 178.5]

vision_figures = [resnet, resnetO, vgg19, vgg19O, regnet, regnetO]
vision_label = ["Resnet101", "Vgg19", "Regnet-y-128-gf"]
nltp_figures = [gpt, gptO, labse, labseO, gpt_xl, gpt_xlO]
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
    plt.ylabel('Speed-up percent')
    plt.title(title) 
    plt.yticks([0,5,10,15,20,25,30,35])
    plt.legend() 
  
draw(vision_label, vision_figures, 1, "Vision")
draw(nltp_label, nltp_figures, 2, "NLTP")
# function to show the plot 
plt.suptitle("TensorFlow-GPU")
plt.show() 