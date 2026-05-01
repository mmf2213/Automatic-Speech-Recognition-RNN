from flask import Flask, render_template, request
import os
from predict import predict_audio

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

# Ensure uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == "":
        return "No selected file"

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    result = predict_audio(file_path)

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)   