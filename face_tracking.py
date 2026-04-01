# Face Recognition script - Trains and performs real-time face recognition
# Requires: Pre-captured face images in the 'faces' folder

import cv2
import os
import numpy as np

# Load the pre-trained SSD (Single Shot Detector) model for face detection
# This uses a ResNet-based architecture trained on COCO dataset
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Path to the directory containing training face images
dataset_path = "faces"

# Data structures to store training data
faces = []           # List to store face images
labels = []          # List to store labels (person IDs) for each face
label_map = {}       # Dictionary mapping names to numeric labels
current_label = 0    # Counter for assigning unique labels

# Load the Haar Cascade classifier for initial face detection in training images
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ========== TRAINING PHASE ==========
# Load and process all images in the faces directory
for file in os.listdir(dataset_path):
    path = os.path.join(dataset_path, file)

    # Read image from file
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract person name from filename (format: name_count.jpg)
    name = file.split("_")[0]

    # Assign a unique numeric label to each new person
    if name not in label_map:
        label_map[name] = current_label
        current_label += 1

    label = label_map[name]

    # Detect faces in the training image using Haar Cascade
    detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Process each detected face region
    for (x, y, w, h) in detected:
        # Extract face region of interest (ROI)
        face_roi = gray[y:y+h, x:x+w]
        # Resize all faces to uniform size (200x200) for consistency
        face_roi = cv2.resize(face_roi, (200, 200))
        faces.append(face_roi)
        labels.append(label)

# Create and train the LBPH (Local Binary Patterns Histograms) face recognizer
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, np.array(labels))

print("Training complete")

# Create reverse mapping from numeric labels back to person names
label_map = {v: k for k, v in label_map.items()}

# ========== RECOGNITION PHASE ==========
# Reinitialize face cascade for runtime detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start video capture from default webcam (camera 0)
cap = cv2.VideoCapture(0)

# Main recognition loop
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (h, w) = frame.shape[:2]

    # Prepare image blob for DNN model (normalize and resize to 300x300)
    # Mean values are BGR format: [104.0, 177.0, 123.0]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    
    # Set input and perform forward pass through SSD network
    net.setInput(blob)
    detections = net.forward()
    
    # Process detections from the neural network
    for i in range(detections.shape[2]):
        # Get confidence score for this detection
        confidence = detections[0, 0, i, 2]
    
        # Only process detections with confidence > 0.6 (60%)
        if confidence > 0.6:
            # Extract bounding box coordinates (normalized to 0-1 range)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
    
            # Clamp coordinates to frame boundaries to avoid out-of-bounds errors
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
    
            # Extract face region of interest (ROI) from grayscale frame
            face_roi = gray[y1:y2, x1:x2]
    
            # Skip if ROI is empty (invalid detection)
            if face_roi.size == 0:
                continue
            
            # Resize face ROI to match training size (200x200)
            face_roi = cv2.resize(face_roi, (200, 200))
    
            # Predict identity and get confidence score from LBPH model
            label, conf = model.predict(face_roi)
            
            # Determine the name to display
            # If confidence is high (>90), mark as "Unknown", otherwise use trained label
            name = label_map[label] if conf < 90 else "Unknown"
    
            # Draw green bounding box around detected face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display the person's name above the bounding box
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with detected and recognized faces
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

