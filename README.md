# SignTalk
## Link for PPT
#### https://www.canva.com/design/DAGPtRlSULQ/uEHuTUTHPhfsd-v-uWuHYw/edit?utm_content=DAGPtRlSULQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
## Overview
This Python script implements a hand gesture recognition system using MediaPipe for hand tracking and a K-Nearest Neighbors (KNN) classifier for gesture recognition. The system allows for both static and dynamic gesture recording, classification, and management.

## Dependencies
- `cv2`: OpenCV library for computer vision tasks.
- `mediapipe`: A framework for building perception pipelines, specifically for hand detection.
- `numpy`: Library for numerical operations.
- `os`: Module for interacting with the operating system.
- `pickle`: Module for serializing and deserializing Python objects.
- `sklearn`: Machine learning library used for KNN classification.

## Global Variables
- `mp_hands`: MediaPipe hands model for hand detection.
- `mp_drawing`: Utility for drawing hand landmarks on the image.
- `GESTURE_PATH`: Path to save static gesture data as a pickle file.
- `DYNAMIC_GESTURE_PATH`: Path to save dynamic gesture data as a pickle file.
- `knn`: KNeighborsClassifier instance for classifying static gestures.

## Data Loading
The code checks if gesture data files exist:
- If `gesture_data.pkl` exists, it loads the landmark data and corresponding labels.
- If `dynamic_gesture_data.pkl` exists, it loads dynamic gesture sequences and their labels.
- If the files do not exist, it initializes empty lists for landmarks and labels.

## Functions

### 1. `train_knn_classifier(X, y)`
Trains the KNN classifier using the provided landmark data (`X`) and their corresponding labels (`y`).
- **Parameters**:
  - `X`: List of landmark data for static gestures.
  - `y`: List of corresponding gesture labels.
- **Output**: Prints the number of samples used for training.

### 2. `classify_gesture_knn(landmarks)`
Classifies a given set of landmarks using the trained KNN classifier.
- **Parameters**:
  - `landmarks`: A list of landmark coordinates to classify.
- **Returns**: The predicted gesture label or an error message if the model is not trained.

### 3. `record_custom_gesture(landmarks, gesture_name)`
Records a new static gesture by appending landmarks and their label to the existing data, then retrains the KNN classifier.
- **Parameters**:
  - `landmarks`: A list of landmark coordinates for the gesture.
  - `gesture_name`: The name of the gesture to be recorded.
- **Output**: Prints a confirmation message and saves the updated gesture data.

### 4. `record_dynamic_gesture(gesture_name, sequence)`
Records a dynamic gesture by appending the sequence of landmarks and the corresponding gesture name to the existing data.
- **Parameters**:
  - `gesture_name`: The name of the dynamic gesture.
  - `sequence`: A list of landmark sequences representing the dynamic gesture.
- **Output**: Prints a confirmation message and saves the updated dynamic gesture data.

### 5. `list_gestures()`
Lists all static and dynamic gestures recorded in the system, avoiding duplicates.

### 6. `delete_gesture(gesture_name)`
Deletes a specified static or dynamic gesture from the recorded data.
- **Parameters**:
  - `gesture_name`: The name of the gesture to delete.
- **Output**: Prints confirmation of deletion and updates the stored data.

## Main Loop
The main loop captures video frames from the webcam and processes them to detect hand landmarks:
- **Key Bindings**:
  - `c`: Capture and save a static gesture.
  - `d`: Start/stop recording a dynamic gesture.
  - `q`: Exit the application.

- When hands are detected:
  - Draws landmarks on the frame.
  - Classifies the current gesture using the KNN classifier.
  - If recording a dynamic gesture, it appends the landmarks to the sequence.

### Webcam Feed
The script starts a webcam feed to continuously capture frames and process hand gestures until the user decides to exit.

## Usage
1. Run the script in an environment with the necessary dependencies installed.
2. Ensure the webcam is functional and accessible.
3. Use the designated keys to interact with the system:
   - Capture static gestures by pressing `c`.
   - Record dynamic gestures by pressing `d`.
   - List all gestures or delete specific gestures as required.
   - Press `q` to exit the application.

This documentation provides a clear understanding of the code's structure and functionality, making it easier to navigate and utilize the gesture recognition system effectively.
## Team members
#### Yashaswi
#### Anubhav
#### Akshat
#### Chetanya
#### Shikhara
