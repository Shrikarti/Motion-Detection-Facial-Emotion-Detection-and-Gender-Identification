# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:46:13 2024

@author: Shrikarti
"""

import cv2

# Load the video
video_path = "C://Users//RADHASHYAM//Downloads//video.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit(1)

# Get the frame rate of the video for timestamp calculations
fps = cap.get(cv2.CAP_PROP_FPS)

# Parameters for motion detection
threshold_value = 20  # Adjust based on sensitivity needed
min_area = 300  # Minimum area to be considered as motion (adjustable)

# Read the first frame
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# To store event timestamps
events = []

while cap.isOpened():
    # Read the next frame
    ret, curr_frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between current and previous frame
    frame_diff = cv2.absdiff(curr_frame_gray, prev_frame_gray)

    # Apply a binary threshold to the difference image
    _, thresh = cv2.threshold(frame_diff, threshold_value, 255, cv2.THRESH_BINARY)

    # Display the thresholded image for debugging
    cv2.imshow('Frame Difference', thresh)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a flag to mark motion
    motion_detected = False

    # Loop through the contours to identify significant motion
    for contour in contours:
        # Ignore small contours based on area
        if cv2.contourArea(contour) < min_area:
            continue

        # Draw bounding box around significant motion
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Mark that motion is detected
        motion_detected = True

    # If significant motion is detected, store the event
    if motion_detected:
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert milliseconds to seconds
        events.append(timestamp)
        print(f"Motion detected at {timestamp:.2f}s")
        cv2.putText(curr_frame, f"Event detected at {timestamp:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the frame with motion detection
    cv2.imshow('Motion Detection', curr_frame)

    # Use a longer wait time if frames disappear too quickly
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Update the previous frame to the current one for the next iteration
    prev_frame_gray = curr_frame_gray

# Release resources
cap.release()
cv2.destroyAllWindows()

# Re-open video to save the output
cap = cv2.VideoCapture(video_path)
output_path = 'C://Users//RADHASHYAM//Downloads//output_with_events.mp4'
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Reprocess to save the annotated frames
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, curr_frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between current and previous frame
    frame_diff = cv2.absdiff(curr_frame_gray, prev_frame_gray)
    _, thresh = cv2.threshold(frame_diff, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    if motion_detected:
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        cv2.putText(curr_frame, f"Event detected at {timestamp:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    out.write(curr_frame)
    prev_frame_gray = curr_frame_gray

# Release resources
cap.release()
out.release()
print(f"Events detected at times: {events}")
