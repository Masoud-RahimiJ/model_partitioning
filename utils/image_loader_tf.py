from tensorflow.keras.preprocessing import image as im
import numpy as np
import os

# img_path = "./dataset/dog.jpg"
# image = im.load_img(img_path, target_size=(224, 224))
# image = im.img_to_array(image)
# image = np.expand_dims(image, axis=0)
# image=np.random.randn(int(os.getenv('BS', 1)), 3, 224, 224)
image=np.random.randn(None, 3, 224, 224)

