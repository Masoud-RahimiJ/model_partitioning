import matplotlib.pyplot as plt 


resnet = [4.15, 4.17, 4.2, 4.26, 4.47, 4.93]
resnetO = [4.26, 4.3, 4.34, 4.45, 4.72, 5.24]

vgg19 = [3.2, 3.21, 3.26, 3.37, 3.53, 3.88]
vgg19O = [3.47, 3.5, 3.58, 3.77, 4.14, 4.81]

regnet = [10.4, 10.65, 11.9, 15.15, 20.6, 31.4]
regnetO = [13.3, 13.7, 14.9, 17.7, 23.2, 33.8]

gpt = [4.8, 4.97, 5.32, 6.05, 7.5, 10.8]
gptO = [5.1, 5.25, 5.55, 6.2, 7.6, 10.9]

labse = [12.65, 12.9, 14.4, 17.7, 23.6, 36]
labseO = [14.8, 15.47, 17, 20.1, 26.1, 38.4]

gpt_xl = [17.6, 18.4, 21, 23.4, 28.4, 39]
gpt_xlO = [22.55, 23.1, 24.25, 26.6, 31.3, 41]

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
    plt.yticks([0,5,10,15,20,25,30])
    plt.legend() 
  
draw(vision_label, vision_figures, 1, "Vision")
draw(nltp_label, nltp_figures, 2, "NLTP")
# function to show the plot 
plt.suptitle("TensorFlow-CPU")
plt.show() 
        
