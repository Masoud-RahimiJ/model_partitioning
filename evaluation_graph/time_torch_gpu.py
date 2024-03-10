import matplotlib.pyplot as plt 


resnet = [3.1, 3.1, 3.1, 3.11, 3.12, 3.13]
resnetO = [4.52, 4.52, 4.53, 4.53, 4.54, 4.55]

vgg19 = [6.38, 6.38, 6.38, 6.39, 6.4, 6.42]
vgg19O = [9.08, 9.08, 9.1, 9.11, 9.13, 9.14]

regnet = [23.73, 23.73, 23.74, 23.76, 23.77, 23.81]
regnetO = [32.6, 32.62, 32.65, 32.72, 32.84, 33.15]

gpt = [9.45, 9.48, 9.55, 9.65, 10, 10.4]
gptO = [11.34, 11.4, 11.55, 11.65, 12.1, 12.8]

labse = [24.4, 24.45, 24.5, 24.6, 24.8, 26.3]
labseO = [25.2, 25.3, 25.4, 25.8, 26.1, 28.9]

gpt_xl = [42.5, 43.3, 46, 49, 55.7, 63]
gpt_xlO = [64, 65, 67, 70, 76, 83]

wav = [9, 9.03, 9.2, 12, 17, 26]
wavO = [12.05, 12.1, 12.5, 15, 20, 30]

whisper_md = [25.7, 25.8, 25.9, 26.3, 26.9, 32.9]
whisper_mdO = [41.6, 41.7, 41.8, 42.2, 42.8, 48.5]

whisper_large = [55, 55.1, 55.5, 56, 60, 70]#, 75]
whisper_largeO = [86.6, 87, 89, 92, 98, 110]#, 116]

"git pull origin gpu && python3 -m evaluation.pytorch.gpt-bse"


vision_figures = [resnet, resnetO, vgg19, vgg19O, regnet, regnetO]
vision_label = ["Resnet101", "Vgg19", "Regnet-y-128-gf"]
nltp_figures = [gpt, gptO, labse, labseO, gpt_xl, gpt_xlO]
nltp_label = ["Gpt2-base", "Labse", "Gpt2-xl"]
voice_figures = [wav, wavO, whisper_md, whisper_mdO, whisper_large, whisper_largeO]
voice_label = ["Wav2vec2-base", "Whisper-medium", "Whisper-large-v2"]


def draw(labels, figures, idx, title):
    plt.subplot(1,3,idx)
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
    plt.yticks([0,10,20,30,40,50,60,70])
    plt.legend() 
  
draw(vision_label, vision_figures, 1, "Vision")
draw(nltp_label, nltp_figures, 2, "NLTP")
draw(voice_label, voice_figures, 3, "Voice")
# function to show the plot 
plt.suptitle("Pytorch-GPU")
plt.show() 
        
