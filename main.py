import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from gtts import gTTS
import tempfile
import playsound

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Path to save gesture data
GESTURE_PATH = "gesture_data.pkl"

# Load saved gestures or create empty list for gestures and labels
if os.path.exists(GESTURE_PATH):
    with open(GESTURE_PATH, 'rb') as f:
        gesture_data = pickle.load(f)
    X, y = gesture_data['landmarks'], gesture_data['labels']
else:
    X, y = [], []

# Initialize StandardScaler for normalization
scaler = StandardScaler()

# Initialize MLPClassifier for static gestures
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

# Variables for gesture recognition
last_recognized_gesture = None  # Track the last recognized gesture
last_recognized_time = 0  # Timestamp of the last recognized gesture
confidence_threshold = 0.98  # Minimum confidence to announce a gesture

# Function to train Neural Network model for static gestures
def train_mlp_classifier(X, y):
    if len(y) < 3:
        print("Not enough data to train the model.")
        return

    # Fit the scaler on the training data and transform the data
    X_scaled = scaler.fit_transform(X)

    # Train the MLP model
    mlp.fit(X_scaled, y)
    print("Model trained with", len(y), "samples.")

# Function to classify static hand gesture using MLP
def classify_gesture_mlp(landmarks):
    landmarks = np.array(landmarks).flatten().reshape(1, -1)  # Reshape to 2D
    try:
        # Ensure scaler is fitted
        check_is_fitted(scaler)

        # Scale the input landmarks
        landmarks_scaled = scaler.transform(landmarks)

        # Check if the MLP model is fitted
        check_is_fitted(mlp)

        # Predict the probabilities
        probabilities = mlp.predict_proba(landmarks_scaled)[0]
        max_confidence_index = np.argmax(probabilities)
        confidence = probabilities[max_confidence_index]
        predicted_label = mlp.classes_[max_confidence_index]
        return predicted_label, confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Function to speak the recognized gesture using gTTS
def speak_gesture(gesture_name):
    try:
        # Generate speech
        tts = gTTS(text=gesture_name, lang='en')
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts.save(temp_audio_file.name)
            # Play the generated audio
            playsound.playsound(temp_audio_file.name)
        # Clean up the temporary file
        os.unlink(temp_audio_file.name)
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

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
    train_mlp_classifier(X, y)

# Start webcam feed
cap = cv2.VideoCapture(0)

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

                # Static gesture recognition using MLP if the model is trained
                recognized_gesture, confidence = classify_gesture_mlp(landmarks)

                # Check if confidence is above threshold and gesture has changed
                if (
                    confidence > confidence_threshold
                    and recognized_gesture != last_recognized_gesture
                ):
                    last_recognized_gesture = recognized_gesture
                    last_recognized_time = time.time()
                    speak_gesture(recognized_gesture)
                    print(f"Recognized: {recognized_gesture} (Confidence: {confidence:.2f})")

                # Display the recognized gesture and confidence on the frame
                cv2.putText(
                    frame,
                    f"{recognized_gesture} ({confidence:.2f})",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

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

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()