from tensorflow.keras.preprocessing import image as im
import numpy as np


img_path = "./dataset/dog.jpg"
image = im.load_img(img_path, target_size=(224, 224))
image = im.img_to_array(image)
image = np.expand_dims(image, axis=0)
