import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('models/imageclassifier.h5')

# Define image preprocessing function
def preprocess_image(image):
    resized = tf.image.resize(image, (256, 256))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            preprocessed_image = preprocess_image(image)
            prediction = model.predict(preprocessed_image)
            predicted_class = "Sad" if prediction > 0.5 else "Happy"
            return render_template('index.html', prediction=predicted_class, image_path=uploaded_file.filename)
    
    # return render_template('index.html', prediction=None, image_path=None)
    return render_template('index.html', prediction=predicted_class, image_path=uploaded_file.filename)


if __name__ == '__main__':
    app.run(debug=True)
