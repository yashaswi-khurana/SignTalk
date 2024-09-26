import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Path to save gesture data
GESTURE_PATH = "gesture_data.pkl"
DYNAMIC_GESTURE_PATH = "dynamic_gesture_data.pkl"

# Load saved gestures or create empty list for gestures and labels
if os.path.exists(GESTURE_PATH):
    with open(GESTURE_PATH, 'rb') as f:
        gesture_data = pickle.load(f)
    X, y = gesture_data['landmarks'], gesture_data['labels']
else:
    X, y = [], []

# Load dynamic gestures or create empty lists for sequences
if os.path.exists(DYNAMIC_GESTURE_PATH):
    with open(DYNAMIC_GESTURE_PATH, 'rb') as f:
        dynamic_gesture_data = pickle.load(f)
    X_dynamic, y_dynamic = dynamic_gesture_data['landmarks'], dynamic_gesture_data['labels']
else:
    X_dynamic, y_dynamic = [], []

# Initialize KNN classifier for static gestures
knn = KNeighborsClassifier(n_neighbors=1)

# Function to train KNN model for static gestures
def train_knn_classifier(X, y):
    n_neighbors = min(3, len(y))  # Ensure n_neighbors is never greater than the number of samples
    knn.set_params(n_neighbors=n_neighbors)
    knn.fit(X, y)
    print("Model trained with", len(y), "samples.")

# Function to classify static hand gesture using KNN
def classify_gesture_knn(landmarks):
    landmarks = np.array(landmarks).flatten().reshape(1, -1)  # Reshape to 2D
    try:
        # Check if the KNN model is fitted
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(knn)
        return knn.predict(landmarks)[0]
    except:
        return "Model not trained yet"

# Function to record new static gestures
def record_custom_gesture(landmarks, gesture_name):
    X.append(landmarks)
    y.append(gesture_name)

    # Save updated gesture data
    gesture_data = {'landmarks': X, 'labels': y}
    with open(GESTURE_PATH, 'wb') as f:
        pickle.dump(gesture_data, f)

    print(f"Custom gesture '{gesture_name}' saved!")
    # Retrain the model with new data
    train_knn_classifier(X, y)

# Function to record dynamic gesture sequences
def record_dynamic_gesture(gesture_name, sequence):
    X_dynamic.append(sequence)
    y_dynamic.append(gesture_name)

    # Save dynamic gesture data
    dynamic_gesture_data = {'landmarks': X_dynamic, 'labels': y_dynamic}
    with open(DYNAMIC_GESTURE_PATH, 'wb') as f:
        pickle.dump(dynamic_gesture_data, f)

    print(f"Dynamic gesture '{gesture_name}' saved!")
    # Load saved gestures or create empty list for gestures and labels
    if os.path.exists(GESTURE_PATH):
        with open(GESTURE_PATH, 'rb') as f:
            gesture_data = pickle.load(f)
        X, y = gesture_data['landmarks'], gesture_data['labels']
    else:
        X, y = [], []

    # Load dynamic gestures or create empty lists for sequences
    if os.path.exists(DYNAMIC_GESTURE_PATH):
        with open(DYNAMIC_GESTURE_PATH, 'rb') as f:
            dynamic_gesture_data = pickle.load(f)
        X_dynamic, y_dynamic = dynamic_gesture_data['landmarks'], dynamic_gesture_data['labels']
    else:
        X_dynamic, y_dynamic = [], []

    # Function to list all gestures
    def list_gestures():
        print("Static Gestures:")
        for gesture in set(y):  # Using set to avoid duplicates
            print(f"- {gesture}")

        print("\nDynamic Gestures:")
        for gesture in set(y_dynamic):  # Using set to avoid duplicates
            print(f"- {gesture}")

    # Function to delete a gesture by name
    def delete_gesture(gesture_name):
        global X, y, X_dynamic, y_dynamic

        # Deleting static gesture
        if gesture_name in y:
            index = y.index(gesture_name)
            del X[index]  # Remove the associated landmarks
            del y[index]  # Remove the gesture name
            print(f"Static gesture '{gesture_name}' deleted.")

            # Save updated gesture data
            gesture_data = {'landmarks': X, 'labels': y}
            with open(GESTURE_PATH, 'wb') as f:
                pickle.dump(gesture_data, f)
        else:
            print(f"Static gesture '{gesture_name}' not found.")

        # Deleting dynamic gesture
        if gesture_name in y_dynamic:
            indices_to_delete = [i for i, name in enumerate(y_dynamic) if name == gesture_name]
            for index in sorted(indices_to_delete, reverse=True):  # Remove in reverse order to avoid indexing issues
                del X_dynamic[index]
                del y_dynamic[index]
            print(f"Dynamic gesture '{gesture_name}' deleted.")

            # Save updated dynamic gesture data
            dynamic_gesture_data = {'landmarks': X_dynamic, 'labels': y_dynamic}
            with open(DYNAMIC_GESTURE_PATH, 'wb') as f:
                pickle.dump(dynamic_gesture_data, f)
        else:
            print(f"Dynamic gesture '{gesture_name}' not found.")

# Start webcam feed
cap = cv2.VideoCapture(0)
sequence = []  # Store dynamic gesture landmarks sequence
recording_dynamic = False  # Flag to indicate dynamic gesture recording
frame_count = 0  # Frame counter for dynamic gestures

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark positions as a flat list
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])

                # Convert to NumPy array
                landmarks = np.array(landmarks).flatten()

                # Static gesture recognition using KNN
                recognized_gesture = classify_gesture_knn(landmarks)
                cv2.putText(frame, recognized_gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # If recording dynamic gesture, append landmarks to the sequence
                if recording_dynamic:
                    sequence.append(landmarks)
                    frame_count += 1
                    cv2.putText(frame, "Recording Dynamic Gesture...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Sign Language Recognition', frame)

        # Press 'c' to capture and create a static gesture
        if cv2.waitKey(1) & 0xFF == ord('c'):
            gesture_name = input("Enter static gesture name: ")
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])
                    landmarks = np.array(landmarks).flatten()
                    record_custom_gesture(landmarks, gesture_name)
            else:
                print("No hand detected to capture gesture.")

        # Press 'd' to start/stop recording dynamic gesture
        if cv2.waitKey(1) & 0xFF == ord('d'):
            if not recording_dynamic:
                gesture_name = input("Enter dynamic gesture name: ")
                sequence = []  # Reset sequence
                recording_dynamic = True
                frame_count = 0
            else:
                recording_dynamic = False
                if frame_count > 0:
                    record_dynamic_gesture(gesture_name, sequence)
                else:
                    print("No frames recorded for dynamic gesture.")

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
