# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:13:54 2024

@author: RADHASHYAM
"""

import cv2
import numpy as np

# Load the image
image_path = "C://Users//RADHASHYAM//Downloads//image.jpg"  
image = cv2.imread(image_path)

# Check if the image is loaded
if image is None:
    print("Error: Could not open or find the image.")
    exit()

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Define the emotions list for labeling
emotions = ["Happy", "Sad", "Thinking", "Surprised", "Angry", "Excited"]

# Function to detect emotions based on facial features
def detect_emotion(face_region):
    height, width = face_region.shape

    # Detect eyes within the face region
    eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

    # Detect mouth within the lower half of the face region
    mouth_region = face_region[int(height * 0.5):, :]
    mouths = mouth_cascade.detectMultiScale(mouth_region, scaleFactor=1.5, minNeighbors=15, minSize=(30, 30))

    # Initialize variables to store emotion features
    mouth_detected = len(mouths) > 0
    eye_count = len(eyes)

    # Analyze mouth aspect ratio for better smile detection
    smile_threshold = 0.5  # Adjust this value to make smile detection more sensitive
    if mouth_detected:
        for (mx, my, mw, mh) in mouths:
            mouth_aspect_ratio = mh / mw  # Ratio of mouth height to width
            if mouth_aspect_ratio < smile_threshold:
                return "Happy"
            elif mouth_aspect_ratio > smile_threshold:
                return "Excited"  # New case for excitement

    # Use conditions to classify emotions based on detected features
    if not mouth_detected and eye_count >= 2:
        if eye_count == 2:
            return "Thinking"  # Renamed from "Neutral"
        elif eye_count > 2:
            return "Surprised"  # Case for wide-open eyes (more eye regions detected)
    elif not mouth_detected and eye_count < 2:
        return "Sad"
    elif eye_count >= 2 and not mouth_detected:
        return "Angry"  # New case for angry (wide-open eyes with no smile)
    else:
        return "Thinking"  # Default to thinking

# Store detected emotions for each face
detected_emotions = []

# Loop through each detected face
for (x, y, w, h) in faces:
    # Draw a rectangle around the face
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Extract the face region for further analysis
    face_region = gray[y:y+h, x:x+w]
    
    # Detect the emotion using the custom function
    emotion = detect_emotion(face_region)
    
    # Store the emotion and draw text
    detected_emotions.append(emotion)
    cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Calculate overall sentiment
overall_sentiment = max(set(detected_emotions), key=detected_emotions.count)

# Display overall sentiment
cv2.putText(image, f"Crowd Sentiment: {overall_sentiment}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

# Save and display the result image
output_path = "C://Users//RADHASHYAM//Downloads//crowd_sentiment.jpg"
cv2.imwrite(output_path, image)
cv2.imshow("Emotion Analysis", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Individual emotions detected: {detected_emotions}")
print(f"Overall crowd sentiment: {overall_sentiment}")
