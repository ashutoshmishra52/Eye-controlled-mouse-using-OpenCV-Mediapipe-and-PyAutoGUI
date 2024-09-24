# Eye Controlled Mouse for Linux - Ashutosh Mishra

import cv2
import mediapipe as mp
import pyautogui

# Disable failsafe to prevent interruptions if the mouse touches the screen corner
pyautogui.FAILSAFE = False

# Set a short delay between mouse movements for smoother operation
pyautogui.PAUSE = 0.2

# Initialize webcam
cam = cv2.VideoCapture(0)

# Initialize face mesh detection with refined landmarks
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get the screen dimensions
screen_w, screen_h = pyautogui.size()

while True:
    # Capture video frame from the webcam
    _, frame = cam.read()
    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)
    # Convert the frame to RGB as Mediapipe requires it
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect face landmarks
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    
    # Get the height and width of the frame
    frame_h, frame_w, _ = frame.shape
    
    if landmark_points:
        landmarks = landmark_points[0].landmark
        # Detect specific face landmarks around the eyes
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            # Draw a circle on detected landmarks (green dots)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Control the mouse pointer based on eye landmarks
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)
        
        # Detect the landmarks for the left eye blink
        left_eye = [landmarks]
