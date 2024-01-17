import matplotlib.pyplot as plt 


resnet = [1.83, 1.89, 1.91, 1.95, 2.06, 2.37]
resnetO = [1.91, 1.98, 2, 2.05, 2.18, 2.47]

vgg19 = [2.46, 2.5, 2.57, 2.64, 2.85, 3.2]
vgg19O = [3.26, 3.4, 3.46, 3.5, 3.76, 4.05]

regnet = [8.5, 8.51, 8.65, 9.45, 11.4, 14.5]
regnetO = [11.1, 11.4, 11.75, 12.12, 13.84, 17.05]

# gpt = [8.21, 9.9, 16.4, 29.1, 56, 109.5]
# gptO = [8.32, 11.3, 18.2, 30.7, 58, 113]
gpt = [5.5, 5.6, 5.7, 5.9, 6.2, 6.5]
gptO = [6.2, 6.3, 6.4, 6.5, 6.7, 7]

# labse = [13.6, 14.6, 17.4, 22.1, 32, 48]
# labseO = [14.6, 17, 18.7, 23.7, 34, 50]
labse = [14.1, 14.2, 14.45, 14.7, 15.3, 16.55]
labseO = [15.9, 16, 16.15, 16.5, 17.2, 18.4]

# gpt_xl = [38.6, 50.1, 84.3, 136, 281, 437]
# gpt_xlO = [51.3, 63.6, 95.9, 149, 296, 456]
gpt_xl = [27.35, 27.7, 28.4, 29.9, 32.1, 36.8]
gpt_xlO = [34, 34.3, 34.7, 35.9, 38.1, 43]


wav = [5.05, 5.5, 6.6, 9, 13.2, 21.8]
wavO = [5.65, 5.9, 7, 9.4, 13.7, 22.4]

whisper_md = [22.9, 24.2, 32.5, 44.4, 62.7, 102]
whisper_mdO = [25.2, 25.9, 34.4, 45.8, 63.8, 103.2]

whisper_large = [43.1, 45.5, 58.8, 78.1, 112.8, 173]
whisper_largeO = [56.8, 60.3, 67.2, 85.5, 117.8, 178.5]

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
    plt.yticks([0,5,10,15,20,25,30,35,40])
    plt.legend() 
  
draw(vision_label, vision_figures, 1, "Vision")
draw(nltp_label, nltp_figures, 2, "NLTP")
draw(voice_label, voice_figures, 3, "Voice")
# function to show the plot 
plt.suptitle("Pytorch-CPU")
plt.show() 
        
