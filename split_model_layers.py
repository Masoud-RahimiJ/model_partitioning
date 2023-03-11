from torch import load, save
from collections import OrderedDict

PATH = "./models/alexnet-owt-4df8aa71"
FORMAT = "pth"

def extract_layer_name(layer):
    return '.'.join(layer.split('.')[0:-1])

def get_layer_file_name(part):
    return PATH + '_' + str(part+1) + '.' + FORMAT

model = load(PATH + '.' + FORMAT)
splitted_model = []
previous_layer_name = ""

for key, value in  model.items():
    layer_name = extract_layer_name(key)    
    if layer_name != previous_layer_name:
        splitted_model.append(OrderedDict())
        splitted_model[-1]._metadata = getattr(model, "_metadata", None)
    splitted_model[-1][key] = value
    previous_layer_name = layer_name
    

for i in range(len(splitted_model)):
    save(splitted_model[i], get_layer_file_name(i))
    
    