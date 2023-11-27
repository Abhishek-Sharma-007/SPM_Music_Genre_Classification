from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os
from werkzeug.utils import secure_filename
import librosa
import librosa.display
app = Flask(__name__)

# Load your model
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



@app.route('/', methods=['GET'])
def index():
    # Render the upload form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Process the file and predict
        processed_data = preprocess_input(file_path)
        processed_data = processed_data.reshape(1, 130, 13)  # Adding batch dimension
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

        # Clean up: Remove the saved file after prediction
        os.remove(file_path)

        return render_template('prediction_result.html', predicted_genre=predicted_genre)


if __name__ == '__main__':
    app.run(debug=True)
