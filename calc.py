import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
import time
import pygame
from PIL import Image

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize pygame mixer for audio playback
click_sound = pygame.mixer.Sound('sample.mp3')

# Parameters
width, height = 1280, 720  # Adjusted resolution for performance
button_radius = 40
space = 20
click_distance = 60
font_scale = 1
font_thickness = 2

# Define calculator buttons and their positions
buttons = [
    '7', '8', '9', '/',
    '4', '5', '6', '*',
    '1', '2', '3', '-',
    '0', '.', '=', '+'
]

# Calculate positions to center circles and text
num_buttons = len(buttons)
rows = 4
cols = 4
total_button_width = cols * 2 * button_radius + (cols - 1) * space
total_button_height = rows * 2 * button_radius + (rows - 1) * space
start_x = (width - total_button_width) // 2
start_y = (height - total_button_height - 2 * button_radius - space) // 2

# Define the circle positions and sizes
circles = [(start_x + col * (2 * button_radius + space) + button_radius,
            start_y + row * (2 * button_radius + space) + button_radius)
           for row in range(rows) for col in range(cols)]

# Additional rectangle on top for displaying input/output
top_rect_width = cols * 2 * button_radius + (cols - 1) * space
top_rect_height = 2 * button_radius
top_rect_position = (start_x, start_y - top_rect_height - space)
top_rect = (top_rect_position, top_rect_width, top_rect_height)

# Define positions for the AC and Backspace buttons
ac_button_position = (start_x + cols * (2 * button_radius + space) + button_radius,
                      start_y + button_radius)
backspace_button_position = (start_x + cols * (2 * button_radius + space) + button_radius,
                             start_y + 3 * (2 * button_radius + space) + button_radius)

# Initialize calculator state
current_input = ""
result = ""
previous_button = None
last_click_time = 0
click_cooldown = 1
button_clicked = None
ac_clicked = False
backspace_clicked = False

# Initialize Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

def process_frame(frame):
    global current_input, result, previous_button, last_click_time, button_clicked, ac_clicked, backspace_clicked

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    hand_landmarks = None
    index_pos = None
    middle_pos = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get finger positions
        index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        index_pos = (int(index_finger.x * width), int(index_finger.y * height))
        middle_pos = (int(middle_finger.x * width), int(middle_finger.y * height))
        
        distance = np.linalg.norm(np.array(index_pos) - np.array(middle_pos))
        
        current_time = time.time()
        
        if distance < click_distance and (current_time - last_click_time) > click_cooldown:
            button_clicked = None
            for (x, y) in circles:
                if (index_pos[0] - x)**2 + (index_pos[1] - y)**2 < button_radius**2:
                    button_clicked = buttons[circles.index((x, y))]
                    break

            # Check for AC button
            if (index_pos[0] - ac_button_position[0])**2 + (index_pos[1] - ac_button_position[1])**2 < button_radius**2:
                current_input = ""
                result = ""
                last_click_time = current_time
                ac_clicked = True
                click_sound.play()
            # Check for Backspace button
            elif (index_pos[0] - backspace_button_position[0])**2 + (index_pos[1] - backspace_button_position[1])**2 < button_radius**2:
                current_input = current_input[:-1]
                last_click_time = current_time
                backspace_clicked = True
                click_sound.play()
            elif button_clicked:
                if button_clicked == "=":
                    if previous_button != "=":
                        try:
                            result = str(eval(current_input))
                        except:
                            result = "Error"
                        current_input = ""
                    previous_button = "="
                else:
                    current_input += button_clicked
                    previous_button = None
                last_click_time = current_time
                click_sound.play()
            else:
                ac_clicked = False
                backspace_clicked = False
                button_clicked = None

    # Draw the circles and add text
    for (x, y) in circles:
        color = (255, 255, 255)
        if button_clicked and buttons[circles.index((x, y))] == button_clicked:
            color = (46, 139, 87)
        cv2.circle(frame, (x, y), button_radius, color, -1)
        cv2.circle(frame, (x, y), button_radius, (0, 0, 0), 2)
        button_text = buttons[circles.index((x, y))]
        text_size, _ = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        cv2.putText(frame, button_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # Draw the top rectangle and display the current input or result
    (tx, ty), tw, th = top_rect
    cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), -1)
    cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 0, 0), 2)
    display_text = result if result else current_input
    text_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_x = tx + (tw - text_size[0]) // 2
    text_y = ty + (th + text_size[1]) // 2
    cv2.putText(frame, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # Draw the distance text
    if index_pos and middle_pos:
        distance_text = f"Distance: {distance:.2f}"
        text_size, _ = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        text_x = 10
        text_y = height - 10
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 10), (text_x + text_size[0], text_y + 10), (128, 128, 128), -1)
        cv2.putText(frame, distance_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw the AC button
    cx, cy = ac_button_position
    color = (255, 255, 255) if not ac_clicked else (46, 139, 87)
    cv2.circle(frame, (cx, cy), button_radius, color, -1)
    cv2.circle(frame, (cx, cy), button_radius, (0, 0, 0), 2)
    cv2.putText(frame, 'AC', (cx - button_radius // 2, cy + button_radius // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # Draw the Backspace button
    cx, cy = backspace_button_position
    color = (255, 255, 255) if not backspace_clicked else (46, 139, 87)
    cv2.circle(frame, (cx, cy), button_radius, color, -1)
    cv2.circle(frame, (cx, cy), button_radius, (0, 0, 0), 2)
    cv2.putText(frame, 'X', (cx - button_radius // 2, cy + button_radius // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    return frame

def calc():
    st.subheader("Hand Gesture Calculator")

    # Create a placeholder for the video frame
    frame_placeholder = st.empty()

    # Capture video from webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    while True:
        success, frame = cap.read()
        if not success:
            st.write("Failed to capture video.")
            break

        frame = process_frame(frame)
        
        # Convert the frame to PIL format for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        # Display the frame in Streamlit
        frame_placeholder.image(frame, use_column_width=True)

        # Add a break condition (for debugging, remove if you want the app to run continuously)
        # if st.button('Stop'):
        #     break

    cap.release()
    st.write("Webcam stopped.")

if __name__ == "__main__":
    calc()
