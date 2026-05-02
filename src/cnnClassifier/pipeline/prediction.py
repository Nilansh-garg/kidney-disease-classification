from unittest import result

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
        result = (raw_prediction[0][0] >= 0.5).astype(int) 
        print(result)
        
        # Get confidence score (probability of predicted class)
        prob = float(raw_prediction[0][0])
        confidence = prob if result == 1 else 1 - prob
        
        if result == 1:
            prediction = "Tumor" # 1 is actually Tumor[cite: 1]
            return [{"class": prediction, "confidence": confidence}]
        else:
            prediction = "Normal" # 0 is actually Normal[cite: 1]
            return [{"class": prediction, "confidence": confidence}]