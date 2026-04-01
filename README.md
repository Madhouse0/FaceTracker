# Face Tracker - Real-Time Face Recognition

A Python-based face recognition system that captures face images from your webcam and performs real-time face detection and recognition using deep learning.

## Features

- **Face Capture**: Easily capture and save face images from your webcam
- **Face Recognition Training**: Train an LBPH face recognizer on your captured faces
- **Real-Time Detection & Recognition**: Identify known faces in real-time using webcam feed
- **Deep Learning-Based Detection**: Uses ResNet SSD model for accurate face detection
- **Multiple Face Support**: Train and recognize multiple different people

## Requirements

- Python 3.6+
- OpenCV (`cv2`)
- NumPy

## Installation

1. Clone or download this repository:
```bash
cd face_tracker_fresher
```

2. Install required dependencies:
```bash
pip install opencv-python numpy
```

## Usage

### Step 1: Capture Face Images

Run the capture script to collect training images:

```bash
python capture.py
```

**Instructions:**
- The webcam feed will open in a window
- Change the `name` variable in `capture.py` to the person's name you're capturing (default is "thang")
- Press **SPACE** to capture a photo (saves to `faces/` folder)
- Press **Q** to quit

> **Tip**: Capture 20-30 images per person from different angles and lighting conditions for best results.

### Step 2: Train and Run Face Recognition

Once you have captured faces for the people you want to recognize, run:

```bash
python face_tracking.py
```

**What it does:**
1. Loads all face images from the `faces/` folder
2. Trains the LBPH face recognizer model
3. Opens the webcam and starts real-time face recognition
4. Displays recognized faces with names and unknown faces as "Unknown"

Press **Q** to exit the application.

## Project Structure

```
face_tracker_fresher/
├── capture.py                              # Script to capture face images
├── face_tracking.py                        # Main face recognition script
├── deploy.prototxt                         # SSD model architecture
├── res10_300x300_ssd_iter_140000.caffemodel  # Pre-trained SSD model weights
├── faces/                                  # Directory to store captured face images
└── README.md                               # This file
```

## How It Works

### Capture Phase
- Uses OpenCV to access the webcam
- Saves captured frames as JPEG images in the `faces/` folder
- Filenames follow the pattern: `{name}_{count}.jpg`

### Training Phase
- Loads all images from the `faces/` folder
- Detects faces using Haar Cascade classifier
- Resizes face regions to 200x200 pixels
- Trains an LBPH (Local Binary Patterns Histograms) face recognizer

### Recognition Phase
- Uses the ResNet SSD deep learning model for face detection (confidence threshold: 0.6)
- Predicts the identity using the trained LBPH model
- Displays bounding boxes and names on recognized faces
- Shows "Unknown" for faces not in the training set (confidence > 90)

## Model Details

- **Face Detection Model**: ResNet SSD (300x300 input size)
- **Face Recognition Model**: LBPH Face Recognizer
- **Supported Input**: Real-time webcam feed

## Tips for Best Results

1. **Image Quality**: Ensure good lighting when capturing faces
2. **Sample Variety**: Capture faces from different angles, distances, and expressions
3. **Consistency**: Use at least 20-30 images per person
4. **Training Time**: Training is fast and happens automatically when running `face_tracking.py`
5. **Confidence Threshold**: Adjust the confidence threshold in `face_tracking.py` (currently 90) to be more or less strict

## Notes

- The `faces/` folder is created automatically if it doesn't exist
- All captured images should be in the `faces/` folder for the model to train properly
- The trained model is not persistent; it retrains each time you run the script
- Unknown faces show a confidence score; adjust the threshold if needed

## License

This project is open source and available for use and modification.
