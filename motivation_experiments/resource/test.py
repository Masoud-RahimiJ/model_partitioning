from tensorflow.keras.applications.vgg19 import decode_predictions, preprocess_input, VGG19
from utils.image_loader_tf import image



OBJECT_NAME="vgg.h5"

model = VGG19(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax", )


model.load_weights(OBJECT_NAME)


image = preprocess_input(image)


preds = model.predict(image)

print('Predicted:', decode_predictions(preds, top=5)[0])
