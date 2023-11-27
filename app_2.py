from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os
from werkzeug.utils import secure_filename
import librosa
import librosa.display
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# Define a dictionary to store username and password pairs (for demonstration purposes)
users = {
    'user1': 'password1',
    'user2': 'password2',
}

# Load your model (consider lazy-loading for larger applications)
model = tf.keras.models.load_model('model_seq_9.h5')

import librosa
import numpy as np
from scipy.ndimage import zoom

# Function to process the input audio file and extract MFCCs
def preprocess_input(file_path, num_mfcc=13, n_fft=2048, hop_length=512):
    """Load an audio file and compute the MFCCs."""
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Compute the MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # Ensure the MFCCs shape is (130, 13)
    mfccs_padded = pad_or_truncate(mfccs, target_shape=(130, 13))

    return mfccs_padded

def pad_or_truncate(mfccs, target_shape):
    # Initialize an array with zeros in the target shape
    mfccs_adjusted = np.zeros(target_shape)

    # Transpose MFCCs to match the target shape format
    mfccs = mfccs.T

    # Determine the number of frames to copy from original MFCCs
    num_frames_original = mfccs.shape[0]  # Number of frames in the original MFCCs
    num_frames_target = target_shape[0]   # Number of frames in the target shape

    # Pad or truncate
    mfccs_adjusted[:min(num_frames_original, num_frames_target), :] = mfccs[:min(num_frames_original, num_frames_target), :]

    return mfccs_adjusted

# Redirect root to login page
@app.route('/')
def root():
    return redirect(url_for('login'))

# Route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users and users[username] == password:
            return redirect(url_for('index'))  # Redirect to the index page upon successful login
        else:
            error_message = "Invalid username or password. Please try again."
            return render_template('login.html', error_message=error_message)

    return render_template('login.html')

# Route for the index page
@app.route('/index')
def index():
    # Render the index page where users can upload and predict audio files
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"

    if file:  # Optional: Add file type validation here
        filename = secure_filename(file.filename)

        # Ensure 'uploads' directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        file_path = os.path.join('uploads', filename)

        try:
            file.save(file_path)

            # Process the file and predict
            processed_data = preprocess_input(file_path)
            processed_data = np.expand_dims(processed_data, axis=0)  # Adding batch dimension
            prediction = model.predict(processed_data)
            predicted_class = np.argmax(prediction, axis=1)

            # Define a custom mapping of class labels to genres
            class_to_genre_mapping = {
                0: "Disco",
                1: "Blues",
                2: "Classical",
                3: "Jazz",
                4: "Country",
                5: "Reggae",
                6: "Hip-Hop",
                7: "Metal",
                8: "Rock",
                9: "Pop"
            }

            # Map the predicted class to the corresponding genre
            predicted_genre = class_to_genre_mapping.get(predicted_class[0], "Unknown")

        except Exception as e:
            print(f"Error during prediction: {e}")
            return f"Error during processing and prediction: {e}"
        finally:
            # Clean up: Remove the saved file after prediction
            if os.path.exists(file_path):
                os.remove(file_path)

        return render_template('prediction_result.html', predicted_genre=predicted_genre)

    # Optional: Return a response for invalid file type
    # return "Invalid file type"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the allowed extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}

if __name__ == '__main__':
    app.run(debug=False)  # Turn off debug mode for production deployment
