import torch
from collections import OrderedDict
from modelClass import AlexNet
import time
from image_loader import image

PATH = "./models/alexnet-owt-4df8aa71"
FORMAT = "pth"
device = torch.device("cpu")

model = AlexNet().to(device)

start_time =time.time()
model_state_dict = torch.load(PATH+"."+FORMAT)
model.load_state_dict(model_state_dict)
model.eval()
output = model.forward(image)
end_time = time.time()
print(end_time-start_time)
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# top5_prob, top5_catid = torch.topk(probabilities, 5)
# with open("imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]
#     for i in range(top5_prob.size(0)):
#         print(categories[top5_catid[i]], top5_prob[i].item())