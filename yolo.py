import os
import cv2
import numpy as np
import torch  # For YOLOv5 model
import mediapipe as mp
import pyttsx3  # Import the pyttsx3 library for text-to-speech
import threading  # Import the threading module to handle concurrent tasks

# Suppress unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize pyttsx3 for voice feedback and set a female voice
engine = pyttsx3.init()
voices = engine.getProperty('voices')

# Set voice index to 1 for female voice if available
if len(voices) > 1:
    engine.setProperty('voice', voices[1].id)
else:
    print("Female voice not found. Using default voice.")

# Function to say text using pyttsx3
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to run speak in a separate thread
def speak_async(text):
    threading.Thread(target=speak, args=(text,)).start()

# Load YOLOv5 model (pretrained on COCO dataset)
try:
    model = torch.hub.load(r'E:\sham\object\yolov5', 'yolov5s', source='local')
    print("YOLOv5 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    exit()

# Initialize OpenCV for webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Initialize MediaPipe for hand position detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for tracking arm positions
landmarks_formed = False
right_hand_raised = False
left_hand_raised = False

# Set window size to fullscreen
cv2.namedWindow("Real-Time Object Detection & Hand Position Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Real-Time Object Detection & Hand Position Detection", cv2.WND_PROP_FULLSCREEN, 1)

# Start MediaPipe Pose for hand position tracking
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        print("Processing frame...")

        # Perform object detection with YOLOv5
        results = model(frame)
        if len(results.xyxy[0]) > 0:  # Check if there are any detections
            for det in results.xyxy[0]:  # Loop through each detection
                x1, y1, x2, y2, conf, cls = det[:6].tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert coordinates to integers
                label = results.names[int(cls)]
                confidence = conf  # Confidence is already in the range [0, 1]

                # Draw the bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Process the pose landmarks
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(image_rgb)

        # If landmarks can be formed (person detected)
        if results_pose.pose_landmarks:
            if not landmarks_formed:
                landmarks_formed = True
                print("Person detected. Begin arm movements.")

            # Store important landmarks for right and left hand detection
            right_shoulder = results_pose.pose_landmarks.landmark[12]
            right_wrist = results_pose.pose_landmarks.landmark[16]
            left_shoulder = results_pose.pose_landmarks.landmark[11]
            left_wrist = results_pose.pose_landmarks.landmark[15]

            # Detect if right hand is raised or lowered
            if right_wrist.y < right_shoulder.y and not right_hand_raised:
                print("Right hand raised")
                speak_async("Right hand raised")  # Use async speak
                right_hand_raised = True
            elif right_wrist.y > right_shoulder.y and right_hand_raised:
                print("Right hand lowered")
                speak_async("Right hand lowered")  # Use async speak
                right_hand_raised = False

            # Detect if left hand is raised or lowered
            if left_wrist.y < left_shoulder.y and not left_hand_raised:
                print("Left hand raised")
                speak_async("Left hand raised")  # Use async speak
                left_hand_raised = True
            elif left_wrist.y > left_shoulder.y and left_hand_raised:
                print("Left hand lowered")
                speak_async("Left hand lowered")  # Use async speak
                left_hand_raised = False

        # If no person detected (no landmarks), indicate that
        elif landmarks_formed:
            landmarks_formed = False
            print("No person detected. Please re-enter the frame.")

        # Show the frame with predictions and pose landmarks
        cv2.imshow("Real-Time Object Detection & Hand Position Detection", frame)

        # Exit loop when 'ESC' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
