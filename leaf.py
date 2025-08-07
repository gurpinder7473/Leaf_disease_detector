from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("model.h5")

class_names = ['Healthy', 'Powdery Mildew', 'Rust', 'Leaf Spot']

def predict_image(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction)]
