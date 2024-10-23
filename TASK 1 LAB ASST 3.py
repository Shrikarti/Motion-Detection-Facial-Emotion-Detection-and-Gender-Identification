# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:53:52 2024

@author: RADHASHYAM
"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
import zipfile

import zipfile
import os

# Path to the zip file
zip_file_path = "D://SEMESTER 7//IAV//face dataset.zip"

# Path to the folder where you want to extract the files
extracted_folder = r"D:\SEMESTER 7\IAV\extracted folder"

# Create the extraction folder if it doesn't exist
os.makedirs(extracted_folder, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

# List the contents of the extracted folder to verify
print("Files extracted to:", extracted_folder)
for root, dirs, files in os.walk(extracted_folder):
    for file in files:
        print(os.path.join(root, file))

subfolder_path = os.path.join(extracted_folder, 'face dataset')
os.listdir(subfolder_path)

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Helper function to detect and crop faces with adjusted parameters
def detect_and_crop_face(image_path):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect frontal faces
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:  # If no frontal face detected, try profile
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
    
    # If face is detected, crop the face
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            return cropped_face, faces
    return None, faces

# Helper function to extract geometric features (distances between landmarks)
def extract_geometric_features(face_image):
    h, w, _ = face_image.shape
    
    # Hypothetical positions of key features (to be replaced with actual landmark detection in a full implementation)
    eye_left = (int(0.3 * w), int(0.4 * h))
    eye_right = (int(0.7 * w), int(0.4 * h))
    nose = (int(0.5 * w), int(0.55 * h))
    mouth = (int(0.5 * w), int(0.75 * h))
    
    # Calculate geometric distances
    eye_distance = np.linalg.norm(np.array(eye_left) - np.array(eye_right))
    nose_to_mouth = np.linalg.norm(np.array(nose) - np.array(mouth))
    face_width = w
    face_height = h
    
    # Add new features: width to height ratio
    width_to_height_ratio = face_width / face_height
    
    return {
        'eye_distance': eye_distance,
        'nose_to_mouth': nose_to_mouth,
        'width_to_height_ratio': width_to_height_ratio
    }

# Helper function to extract texture features using Local Binary Patterns (LBP)
def extract_texture_features(face_image):
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_face, P=8, R=1, method='uniform')
    
    # Calculate histogram of LBP as a texture descriptor
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, 11),
                             range=(0, 10))
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    
    # Add variance of the LBP histogram to capture texture smoothness
    lbp_variance = np.var(hist)
    
    return hist, lbp_variance

# Helper function to classify gender based on geometric and texture features
def classify_gender(geometric_features, texture_features, lbp_variance):
    eye_distance = geometric_features['eye_distance']
    nose_to_mouth = geometric_features['nose_to_mouth']
    width_to_height_ratio = geometric_features['width_to_height_ratio']
    
    # Define a threshold for the eye distance to nose-to-mouth ratio
    eye_to_mouth_ratio = eye_distance / (nose_to_mouth + 1e-6)
    
    # Start with a balanced score system
    score = 0
    
    # Adjust score based on geometric ratios
    if 1.1 <= eye_to_mouth_ratio <= 1.25:
        score += 1  # Slightly more likely to be male
    elif eye_to_mouth_ratio < 1.1:
        score -= 1  # Slightly more likely to be female
    
    # Adjust based on face shape (width-to-height ratio)
    if 0.85 <= width_to_height_ratio <= 1.1:
        score -= 1  # More likely to be female for narrower faces
    elif width_to_height_ratio > 1.1:
        score += 1  # More likely to be male for wider faces
    
    # Refine based on texture analysis:
    smooth_texture_score = np.sum(texture_features[:3])  # Sum up smooth texture patterns
    
    # Texture-based adjustment
    if smooth_texture_score > 0.4:
        score -= 1  # More likely to be female for smoother textures
    if lbp_variance < 0.025:
        score -= 1  # More likely to be female if variance is low (smoother skin)
    elif lbp_variance > 0.03:
        score += 1  # More likely to be male for rougher textures

    # Determine gender based on final score
    if score > 0:
        return 'Male'
    else:
        return 'Female'

sample_image_path = os.path.join(subfolder_path, '090544.jpg.jpg')

# Detect and crop the face from the sample image
cropped_face, face_rect = detect_and_crop_face(sample_image_path)

# Display the original image with face detection and the cropped face (if detected)
if cropped_face is not None:
    # Show original image with face rectangle
    img = cv2.imread(sample_image_path)
    for (x, y, w, h) in face_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Detected Face')
    
    # Show cropped face
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Face')
    
    plt.show()
else:
    print("No face detected in the sample image.")
# Extract features from the cropped face (if face was detected)
if cropped_face is not None:
    geometric_features = extract_geometric_features(cropped_face)
    texture_features, lbp_variance = extract_texture_features(cropped_face)

    # Classify gender using the extracted features
    predicted_gender = classify_gender(geometric_features, texture_features, lbp_variance)

    # Display the result
    print(f"Predicted Gender: {predicted_gender}")
else:
    print("No face detected to classify.")
