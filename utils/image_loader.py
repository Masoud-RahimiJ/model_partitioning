from PIL import Image
import torch
# from torchvision import transforms
# input_image = Image.open("./dataset/dog.jpg")
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(input_image)
# image = input_tensor.unsqueeze(0)
image=torch.rand(1, 3, 224, 224)