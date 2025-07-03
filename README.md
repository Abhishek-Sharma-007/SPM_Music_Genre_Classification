# Music Genre Classification using Deep Learning

## 🎵 Project Overview

This project implements an advanced **Music Genre Classification System** using deep learning techniques to automatically classify audio files into 10 different music genres. The system leverages state-of-the-art convolutional neural networks and audio feature extraction methods to achieve high accuracy in genre prediction.

## 🎯 Supported Genres

The system can classify music into the following 10 genres:
- **Blues**
- **Classical** 
- **Country**
- **Disco**
- **Hip-Hop**
- **Jazz**
- **Metal**
- **Pop**
- **Reggae**
- **Rock**

## 🗂️ Dataset

The project uses the **GTZAN Dataset**, a widely-used benchmark dataset in music information retrieval research:
- **1,000 audio tracks** (100 tracks per genre)
- **30-second audio clips** sampled at 22.05 kHz
- **WAV format** for high-quality audio processing
- Balanced dataset ensuring equal representation of all genres

## 🧠 Machine Learning Models

### 1. **Custom Sequential Model** (`model_seq_9.h5`)
- Base model with MFCC feature extraction
- Dense layers with regularization and dropout
- Optimized for MFCC input shape (130, 13)

### 2. **VGG16 Transfer Learning** (`best_vgg16_model.h5`)
- Pre-trained VGG16 architecture adapted for audio classification
- Input resized to (128, 128, 3) to match VGG16 requirements
- Transfer learning approach for improved performance

### 3. **ResNet50 Architecture** (`best_resnet50_model.h5`)
- Deep residual network for complex pattern recognition
- Global average pooling and custom dense layers
- Advanced regularization techniques

### 4. **DenseNet201 Model** (`best_densenet_model.h5`)
- Dense connectivity pattern for feature reuse
- Efficient parameter usage and gradient flow
- High-capacity model for complex audio patterns

## 🔧 Technical Implementation

### Audio Feature Extraction
- **MFCC (Mel-frequency Cepstral Coefficients)**: Primary audio features
- **Segment-based Processing**: 10 segments per track for data augmentation
- **Feature Shape**: (130, 13) representing time frames and MFCC coefficients
- **Preprocessing**: Normalization and padding/truncation for consistent input

### Data Processing Pipeline
```python
# Audio Loading
audio, sr = librosa.load(file_path, sr=22050)

# MFCC Extraction
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)

# Preprocessing for CNN Models
processed_data = cv2.resize(mfccs, (128, 128))
processed_data = np.stack((processed_data,)*3, axis=-1)  # Convert to 3-channel
```

## 🌐 Web Application

### Flask-based Interface
The project includes multiple Flask applications:

#### **app.py** - Basic Classification Interface
- Simple file upload and prediction
- Real-time genre classification
- Clean, responsive UI

#### **app_2.py** - Enhanced Interface with Authentication
- User login system
- Enhanced error handling
- Secure file upload validation
- Support for multiple audio formats (WAV, MP3, FLAC)

#### **app_vgg16.py** - VGG16 Model Interface
- Specialized interface for VGG16 model
- Optimized preprocessing for transfer learning approach

### Frontend Features
- **Modern UI Design**: Clean, animated interface using CSS3
- **Responsive Layout**: Mobile-friendly design
- **File Upload**: Drag-and-drop audio file upload
- **Real-time Prediction**: Instant genre classification results
- **Material Design Icons**: Professional UI elements

## 📊 Model Performance

### Training Configuration
- **Optimizer**: Adam (learning_rate=0.0005)
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Callbacks**: 
  - ReduceLROnPlateau for adaptive learning
  - EarlyStopping to prevent overfitting
  - ModelCheckpoint for best model saving

### Regularization Techniques
- **L2 Regularization**: Weight decay to prevent overfitting
- **Dropout Layers**: Random neuron deactivation (30% rate)
- **BatchNormalization**: Stable and faster training
- **LeakyReLU Activation**: Improved gradient flow

## 🛠️ Installation and Setup

### Prerequisites
```bash
# Python packages
pip install tensorflow
pip install librosa
pip install flask
pip install numpy
pip install scikit-learn
pip install opencv-python
pip install matplotlib

# System dependencies (Ubuntu/Debian)
sudo apt install ffmpeg
sudo apt install gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

### Running the Application
```bash
# Basic interface
python app.py

# Enhanced interface with authentication
python app_2.py

# VGG16 model interface
python app_vgg16.py
```

## 📁 Project Structure

```
├── app.py                          # Basic Flask application
├── app_2.py                        # Enhanced Flask app with auth
├── app_vgg16.py                    # VGG16 model interface
├── Music_Genre_Classification_research.ipynb  # Main research notebook
├── resnet50.ipynb                  # ResNet50 implementation
├── Untitled.ipynb                  # Additional experiments
├── models/
│   ├── model_seq_9.h5             # Base sequential model
│   ├── best_vgg16_model.h5        # VGG16 transfer learning model
│   ├── best_resnet50_model.h5     # ResNet50 model
│   └── best_densenet_model.h5     # DenseNet201 model
├── templates/
│   ├── index.html                 # Main upload interface
│   ├── login.html                 # Authentication page
│   └── prediction_result.html     # Results display
├── static/
│   ├── css/                       # Stylesheets
│   └── images/                    # UI assets
└── Lab/                           # Research documentation
```

## 🎨 User Interface

### Upload Interface
- Clean, modern design with animations
- Drag-and-drop file upload functionality
- Support for common audio formats
- Real-time file validation

### Results Display
- Clear genre prediction display
- Confidence scores (where applicable)
- Option to upload another file
- Responsive design for all devices

## 🔬 Research and Development

### Experimental Work
- Multiple architecture comparisons
- Hyperparameter optimization
- Data augmentation techniques
- Feature engineering experiments

### Documentation
- Comprehensive research notebooks
- Model performance analysis
- Architecture comparison studies
- Software project management documentation

## 👥 Contributors

This project represents collaborative work in machine learning and software engineering, with contributions to:
- Deep learning model development
- Web application design
- Audio processing pipeline
- User interface development
- Project management and documentation

## 📄 License

This project is developed for educational and research purposes. Please ensure proper attribution when using or extending this work.

---

**Note**: This project demonstrates advanced machine learning techniques for audio classification and serves as a comprehensive example of end-to-end ML system development from research to deployment.
