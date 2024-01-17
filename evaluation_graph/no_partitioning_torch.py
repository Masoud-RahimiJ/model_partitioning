import matplotlib.pyplot as plt 


resnet = [1.83, 1.89, 1.91, 1.95, 2.06, 2.37]
resnet_noP = [4.02, 4.03, 4.05, 4.1, 4.15, 4.3]

vgg19 = [2.46, 2.5, 2.57, 2.64, 2.85, 3.2]
vgg19_noP = [2.61, 2.71, 2.81, 2.9, 3.01, 3.38]

regnet = [8.5, 8.51, 8.65, 9.45, 11.4, 14.5]
regnet_noP = [12.51, 12.61, 12.8, 13.5, 15.9, 18.2]

gpt = [5.5, 5.6, 5.7, 5.9, 6.2, 6.5]
gpt_noP = [5.8, 5.85, 5.9, 6.05, 6.35, 6.73]

labse = [14.1, 14.2, 14.45, 14.7, 15.3, 16.55]
labse_noP = [14.3, 14.45, 14.92, 15, 15.5, 16.71]

gpt_xl = [27.35, 27.7, 28.4, 29.9, 32.1, 36.8]
gpt_xl_noP = [28.4, 28.75, 29.5, 31, 33.2, 38]

wav = [5.05, 5.5, 6.6, 9, 13.2, 21.8]
wav_noP = [5.1, 5.66, 6.8, 9.2, 13.59, 22.56]

whisper_md = [22.9, 24.2, 32.5, 44.4, 62.7, 102]
whisper_md_noP = [24.1, 25.46, 32.94, 45.2, 63, 103]

whisper_large = [43.1, 45.5, 58.8, 78.1, 112.8, 173]
whisper_large_noP = [49.2, 51.06, 67.77, 83.7, 114, 175.7]

"git pull origin gpu && python3 -m evaluation.pytorch.gpt-bse"


vision_figures = [resnet, resnet_noP, vgg19, vgg19_noP, regnet, regnet_noP]
vision_label = ["Resnet101", "Vgg19", "Regnet-y-128-gf"]
nltp_figures = [gpt, gpt_noP, labse, labse_noP, gpt_xl, gpt_xl_noP]
nltp_label = ["Gpt2-base", "Labse", "Gpt2-xl"]
voice_figures = [wav, wav_noP, whisper_md, whisper_md_noP, whisper_large, whisper_large_noP]
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
        
