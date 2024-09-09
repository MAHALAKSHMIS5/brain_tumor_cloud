from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file types (images only)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the trained model
model = load_model('Brain Tumor Classification.h5')

# Define labels
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Check if uploaded file is allowed (correct extension)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image
        img = image.load_img(file_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = labels[np.argmax(prediction)]
        
        # Use url_for to create the correct URL for the uploaded image
        img_url = url_for('uploaded_file', filename=filename)

        return render_template('result.html', prediction=predicted_class, img_url=img_url)
    
    return redirect(request.url)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
