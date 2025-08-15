from flask import Flask, render_template, request, send_from_directory
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model setup
MODEL_PATH = 'dr_model.h5'
MODEL_URL = "https://drive.google.com/uc?id=11DVFnbesDNaxrqCSAwgC8t76Ntgdu2q_"  # your file ID

# Download model from Google Drive if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Download complete.")

model = load_model(MODEL_PATH)
classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Image preprocessing
        img = Image.open(filepath).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        pred_class = classes[np.argmax(preds)]
        confidence = float(np.max(preds)) * 100

        return render_template('index.html',
                               prediction=pred_class,
                               confidence=f"{confidence:.2f}",
                               filename=filename)
    return "Something went wrong"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
