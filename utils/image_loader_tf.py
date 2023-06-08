from tensorflow.keras.preprocessing import image
import numpy as np


img_path = "./dataset/dog.jpg"
image = image.load_img(img_path, target_size=(224, 224))
image = image.img_to_array(image)
image = np.expand_dims(image, axis=0)
