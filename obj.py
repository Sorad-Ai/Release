import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

def obj():
    st.subheader("YOLO Object Detection")

    # Load the YOLO model
    model = YOLO("./yolov8x.pt")

    # Create a placeholder for the video feed
    video_placeholder = st.empty()

    # Open webcam feed
    cap = cv2.VideoCapture(0)

    # Set the desired resolution (HD)
    width, height = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video.")
            break

        # Perform object detection
        results = model.predict(source=frame, show=False)

        # Draw results on frame
        annotated_frame = results[0].plot()

        # Convert frame to RGB and resize to HD resolution
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(annotated_frame, (width, height))
        image = Image.fromarray(resized_frame)

        # Show the frame in the Streamlit app
        video_placeholder.image(image, channels="RGB", use_column_width=True)

    cap.release()
    st.write("Webcam stopped.")

if __name__ == "__main__":
    main()
