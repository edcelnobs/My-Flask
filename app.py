import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load the pre-trained Keras model
model_path = 'C:/Users/Edz/Desktop/Keras-mod/mod2/modelp.keras'
model = load_model(model_path)

# Define the categories
data_cat = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
img_height, img_width = 180, 180

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and preprocess the image for prediction
            with Image.open(filepath) as image_load:
                image_load = image_load.resize((img_height, img_width))
                img_arr = np.array(image_load)
                img_bat = np.expand_dims(img_arr, axis=0)
                
                # Ensure the image has 3 color channels
                if img_bat.shape[-1] == 1:
                    img_bat = np.repeat(img_bat, 3, axis=-1)
                elif img_bat.shape[-1] != 3:
                    raise ValueError("Image must have 1 or 3 color channels")
                
                # Make predictions
                predict = model.predict(img_bat)
                score = tf.nn.softmax(predict[0])

                # Display the prediction result
                result = {
                    'filename': filename,
                    'prediction': data_cat[np.argmax(score)],
                    'accuracy': f'{np.max(score) * 100:.2f}%'
                }
                return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
