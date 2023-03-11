import torch
from collections import OrderedDict
from modelClass import AlexNet
import time
from image_loader import image
import threading



PATH = "./models/alexnet-owt-4df8aa71"
FORMAT = "pth"
LAYER_COUNT = 8
device = torch.device("cpu")

model = AlexNet().to(device)


def get_layer_file_name(part):
    return PATH + '_' + str(part+1) + '.' + FORMAT

def forward_pre_hook(module, input):
    if module.must_be_loaded:
        module.is_loaded_lock.acquire()
        module.is_loaded_lock.release()


def load_model():
    for i in range(LAYER_COUNT):
        file_name = get_layer_file_name(i)
        layer = torch.load(file_name)
        model.load_state_dict(layer, strict=False)
        model.eval()


torch.nn.modules.module.register_module_forward_pre_hook(forward_pre_hook)
start_time =time.time()
model_loading_thread = threading.Thread(target=load_model)
model_loading_thread.start()
output = model.forward(image)
end_time =time.time()
print(end_time-start_time)
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# top5_prob, top5_catid = torch.topk(probabilities, 5)
# with open("imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]
#     for i in range(top5_prob.size(0)):
#         print(categories[top5_catid[i]], top5_prob[i].item())