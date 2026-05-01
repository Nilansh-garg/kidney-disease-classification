from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import predict_pipeline

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = predict_pipeline(self.filename)

# Home page route
@app.route("/", methods=['GET'])
@cross_origin()
def homePage():
    return render_template("index.html")

# Classification page route
@app.route("/classify", methods=['GET'])
@cross_origin()
def classifyPage():
    return render_template("classify.html")

# About page route
@app.route("/about", methods=['GET'])
@cross_origin()
def aboutPage():
    return render_template("about.html")

# Contact page route
@app.route("/contact", methods=['GET'])
@cross_origin()
def contactPage():
    return render_template("contact.html")

# Training route
@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    try:
        os.system("dvc repro")
        return jsonify({"status": "success", "message": "Training successful"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Prediction route
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)
        result = clApp.classifier.predict()
        return jsonify(result)
    except KeyError:
        return jsonify({"status": "error", "message": "No image data provided"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# API health check
@app.route("/health", methods=['GET'])
@cross_origin()
def healthCheck():
    return jsonify({"status": "healthy", "message": "API is running"})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template("index.html"), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": "error", "message": "Internal server error"}), 500

if __name__ == "__main__":
    clApp = ClientApp()
    # For production, set debug=False
    app.run(host="0.0.0.0", port=7860, debug=True)