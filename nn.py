import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

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

# Initialize StandardScaler for normalization
scaler = StandardScaler()

# Initialize Neural Network for static gestures
model = None

# Function to create and compile a neural network model
def create_neural_network(input_shape, num_classes):
    global model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # One output per class
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

# Function to train the neural network for static gestures
def train_neural_network(X, y):
    if len(y) < 3:
        print("Not enough data to train the model.")
        return

    # Fit the scaler on the training data and transform the data
    X_scaled = scaler.fit_transform(X)

    # Create and compile the neural network if it doesn't exist
    num_classes = len(np.unique(y))  # Number of unique gestures
    if model is None:
        create_neural_network(input_shape=(X_scaled.shape[1],), num_classes=num_classes)

    # Train the neural network
    y = np.array(y)  # Ensure labels are in NumPy array format
    model.fit(X_scaled, y, epochs=50, batch_size=8, verbose=1)
    print("Neural network trained with", len(y), "samples.")

# Function to classify static hand gesture using the neural network
def classify_gesture_nn(landmarks):
    landmarks = np.array(landmarks).flatten().reshape(1, -1)  # Reshape to 2D
    try:
        # Ensure scaler is fitted
        check_is_fitted(scaler)

        # Scale the input landmarks
        landmarks_scaled = scaler.transform(landmarks)

        # Predict gesture with the trained neural network
        predictions = model.predict(landmarks_scaled)
        predicted_class = np.argmax(predictions, axis=1)[0]
        return predicted_class
    except Exception as e:
        return f"Error: {str(e)}"

# Function to record new static gestures
def record_custom_gesture(landmarks, gesture_name):
    X.append(landmarks)
    y.append(gesture_name)

    # Save updated gesture data
    gesture_data = {'landmarks': X, 'labels': y}
    with open(GESTURE_PATH, 'wb') as f:
        pickle.dump(gesture_data, f)

    print(f"Custom gesture '{gesture_name}' saved!")

    # Retrain the neural network with new data
    train_neural_network(X, y)

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

                # Static gesture recognition using Neural Network
                recognized_gesture = classify_gesture_nn(landmarks)
                cv2.putText(frame, str(recognized_gesture), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Sign Language Recognition', frame)

        # Press 'c' to capture and create a static gesture
        if cv2.waitKey(1) & 0xFF == ord('c'):
            gesture_name = int(input("Enter static gesture label (integer): "))
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
