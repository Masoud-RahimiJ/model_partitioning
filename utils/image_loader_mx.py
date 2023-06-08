import mxnet as mx
import numpy as np

img_path = "./dataset/dog.jpg"
image = mx.image.imread(img_path)
image = mx.image.resize_short(image, 224) #minimum 224x224 images
image, _ = mx.image.center_crop(image, (224, 224))
image = mx.image.color_normalize(image.astype(np.float32)/255,
                                    mean=mx.nd.array([0.485, 0.456, 0.406]),
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
image = image.transpose((2,0,1))  
image = image.expand_dims(axis=0)