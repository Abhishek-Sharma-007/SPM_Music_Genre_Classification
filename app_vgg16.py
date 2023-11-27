from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
import librosa
import librosa.display
from scipy.ndimage import zoom
from skimage.transform import resize
app = Flask(__name__)

# Load your new model based on VGG16 with three channels
new_model = tf.keras.models.load_model('best_vgg16_model.h5')

# Function to preprocess the input audio file and resize it to match VGG16 input shape
def preprocess_input(file_path, target_shape=(128, 128)):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Compute the MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Resize the MFCCs to the target shape using scikit-image
    mfccs_resized = resize(mfccs, target_shape, mode='constant')

    # Stack the MFCCs along the third axis to create 3 channels
    mfccs_resized = np.stack((mfccs_resized,) * 3, axis=-1)

    return mfccs_resized

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
        processed_data = processed_data.reshape(1, 128, 128, 3)  # Reshape for the VGG16-based model

        # Make a prediction using the VGG16-based model
        prediction = new_model.predict(processed_data)
        predicted_class_index = np.argmax(prediction, axis=1)

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
        predicted_genre = class_to_genre_mapping.get(predicted_class_index[0], "Unknown")

        # Clean up: Remove the saved file after prediction
        os.remove(file_path)

        return render_template('prediction_result.html', predicted_genre=predicted_genre)
if __name__ == '__main__':
    app.run(debug=True)
