# MUSIC GENRE CLASSIFICATION

A deep learning-based music genre classification system that can automatically classify audio files into 10 different music genres using various neural network architectures including Sequential models, VGG16, ResNet50, and DenseNet201.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [API Endpoints](#api-endpoints)
- [File Structure](#file-structure)
- [Technical Details](#technical-details)
- [Performance](#performance)
- [Contributors](#contributors)
- [License](#license)

## Overview

This project implements a comprehensive music genre classification system that uses Mel-frequency cepstral coefficients (MFCCs) as audio features and deep learning models for classification. The system includes:

- Multiple deep learning models for genre classification
- Web-based interface for audio file upload and prediction
- Support for various audio formats
- Real-time classification with high accuracy
- User authentication system

## Features

- **Multi-Model Support**: Implements multiple deep learning architectures (Sequential CNN, VGG16, ResNet50, DenseNet201)
- **Web Interface**: Clean and modern UI for easy audio file upload
- **10 Genre Classification**: Supports Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, and Rock
- **Audio Preprocessing**: Automatic MFCC extraction and feature engineering
- **User Authentication**: Login system for access control (in app_2.py)
- **Real-time Prediction**: Fast inference with pre-trained models
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

### System Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Web Browser   │────▶│   Flask Server  │────▶│   ML Models     │
│   (Frontend)    │     │   (Backend)     │     │   (TensorFlow)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                        │
         │                       │                        │
         ▼                       ▼                        ▼
    HTML/CSS/JS            Python/Flask            .h5 Model Files
```

### Model Architecture
- **Sequential Model**: Custom CNN with multiple convolutional layers
- **VGG16**: Transfer learning with ImageNet pre-trained weights
- **ResNet50**: Transfer learning for robust feature extraction
- **DenseNet201**: Dense connections for efficient parameter usage

## Dataset

The project uses the GTZAN Genre Collection dataset, which contains:
- 1000 audio tracks (100 tracks per genre)
- 30-second audio clips
- 22050 Hz sampling rate
- 10 music genres

### Data Processing Pipeline:
1. Audio loading using librosa
2. MFCC extraction (13 coefficients)
3. Segmentation (10 segments per track)
4. Feature normalization
5. Reshaping for CNN input

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg for audio processing

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/music-genre-classification.git
cd music-genre-classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

5. Create required directories:
```bash
mkdir uploads
```

## Usage

### Running the Web Application

1. Basic application (no authentication):
```bash
python app.py
```

2. Application with login system:
```bash
python app_2.py
```

3. VGG16 model application:
```bash
python app_vgg16.py
```

4. Access the application:
```
http://localhost:5000
```

### Using the Web Interface
1. Navigate to the home page
2. Click "Upload Audio File"
3. Select an audio file (WAV, MP3, or FLAC)
4. Click "Upload and Predict"
5. View the predicted genre

### Training Models

To train models from scratch, use the Jupyter notebooks:
```bash
jupyter notebook Music_Genre_Classification_research.ipynb
```

## Models

### Pre-trained Models Available:
1. **model_seq_9.h5** (31MB) - Sequential CNN model
2. **best_vgg16_model.h5** (81MB) - VGG16 transfer learning
3. **best_resnet50_model.h5** (97MB) - ResNet50 transfer learning
4. **best_densenet_model.h5** (31MB) - DenseNet201 model

### Model Performance:
| Model | Validation Accuracy | Training Time |
|-------|-------------------|---------------|
| Sequential CNN | ~85% | 2 hours |
| VGG16 | ~89% | 3 hours |
| ResNet50 | ~91% | 4 hours |
| DenseNet201 | ~88% | 3.5 hours |

## API Endpoints

### Main Application Routes:
- `GET /` - Home page
- `GET /login` - Login page (app_2.py)
- `POST /login` - Authentication endpoint
- `GET /index` - Main upload interface
- `POST /predict` - Audio classification endpoint

### Request Format for `/predict`:
```
POST /predict
Content-Type: multipart/form-data
Body: audio file (WAV/MP3/FLAC)
```

### Response Format:
```html
Rendered HTML page with predicted genre
```

## File Structure

```
music-genre-classification/
│
├── app.py                  # Main Flask application
├── app_2.py               # Flask app with authentication
├── app_vgg16.py           # VGG16 model application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
│
├── models/                # Saved model files
│   ├── model_seq_9.h5
│   ├── best_vgg16_model.h5
│   ├── best_resnet50_model.h5
│   └── best_densenet_model.h5
│
├── templates/             # HTML templates
│   ├── index.html
│   ├── login.html
│   └── prediction_result.html
│
├── static/                # Static assets
│   ├── css/
│   └── images/
│
├── notebooks/             # Jupyter notebooks
│   ├── Music_Genre_Classification_research.ipynb
│   ├── resnet50.ipynb
│   └── Untitled.ipynb
│
├── Lab/                   # Research documents
│   └── various PDFs
│
└── uploads/              # Temporary file storage
```

## Technical Details

### Audio Feature Extraction:
```python
# MFCC extraction parameters
num_mfcc = 13
n_fft = 2048
hop_length = 512
sample_rate = 22050
```

### Data Preprocessing:
```python
# Input shape for models
Sequential: (130, 13)
VGG16/ResNet50: (128, 128, 3)
DenseNet201: (224, 224, 3)
```

### Genre Mapping:
```python
{
    0: "Blues",
    1: "Classical", 
    2: "Country",
    3: "Disco",
    4: "Hip-Hop",
    5: "Jazz",
    6: "Metal",
    7: "Pop",
    8: "Reggae",
    9: "Rock"
}
```

## Performance

### System Requirements:
- **Minimum**: 4GB RAM, Intel i3 processor
- **Recommended**: 8GB RAM, NVIDIA GPU with 4GB VRAM
- **Storage**: 500MB for models and application

### Inference Speed:
- Sequential Model: ~0.5 seconds per prediction
- VGG16: ~0.8 seconds per prediction
- ResNet50: ~1.0 seconds per prediction
- DenseNet201: ~0.9 seconds per prediction

## Contributors

This project was developed as part of the Abhishek branch of the music genre classification project. Initial changes were made to the Abhishek Branch.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

### Quick Start Commands

```bash
# Clone and setup
git clone <repository-url>
cd music-genre-classification
pip install -r requirements.txt

# Run the application
python app.py

# Access at http://localhost:5000
```

For more information or issues, please check the GitHub repository or contact the maintainers.
