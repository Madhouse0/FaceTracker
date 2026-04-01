# Capture script to collect face training images from webcam
# Run this script first to gather Face images for training

import cv2
import os

# Create faces directory if it doesn't exist (stores training images)
os.makedirs("faces", exist_ok=True)

# Initialize webcam capture (0 is the default camera)
cap = cv2.VideoCapture(0)
count = 0
name = "thang"  # Change this to the person's name you are capturing

# Main capture loop
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Display the webcam feed with instructions
    cv2.imshow("Capture - Press SPACE to take photo", frame)

    # Wait for keyboard input (1ms timeout)
    key = cv2.waitKey(1)

    # Press SPACE (key code 32) to capture a photo
    if key == 32:
        filename = f"faces/{name}_{count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        count += 1

    # Press 'q' to quit the application
    elif key == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()