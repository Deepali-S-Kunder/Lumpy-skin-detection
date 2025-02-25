from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import BinaryCrossentropy

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = 'CNN.h5'

# Overriding the loss function while loading
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])

# Define a folder to save uploaded images inside the static directory
UPLOAD_FOLDER = os.path.join('static', 'uploads')  # Save in static/uploads
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image file is uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image
            img = image.load_img(filepath, target_size=(224, 224))  # Adjust size based on model
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

            # Predict using the model
            prediction = model.predict(img_array)
            result = 'Lumpy Skin Disease Detected' if prediction[0][0] > 0.5 else 'No Lumpy Skin Disease'

            # Generate the relative URL for the uploaded image
            image_url = url_for('static', filename=f'uploads/{filename}')
            return render_template('index.html', result=result, image_path=image_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
