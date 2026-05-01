import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class predict_pipeline:
    def __init__(self, filename):
        self.filename = filename
        
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training","model.h5"))
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image,axis = 0)
        raw_prediction = model.predict(test_image)
        print(raw_prediction)
        result = np.argmax(raw_prediction,axis = 1)
        print(result)
        
        # Get confidence score (probability of predicted class)
        confidence = float(np.max(raw_prediction))
        
        if result[0] == 1:
            prediction = "Normal"
            return [{"class": prediction, "confidence": confidence}]
        else:
            prediction = "Tumor"
            return [{"class": prediction, "confidence": confidence}]