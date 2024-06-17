import streamlit as st
import base64
import cv2
import numpy as np
import asyncio
from detection import AccidentDetectionModel  # Ensure this import works correctly with your local setup
from concurrent.futures import ThreadPoolExecutor
import time
from notificationapi_python_server_sdk import notificationapi
import csv

# Define headers for the notification API
headers = {
    "clientId": st.secrets["clientId"],
    "clientSecret": st.secrets["clientSecret"],
    "email": st.secrets["email"],
    "number": st.secrets["number"],
    "content-type": "application/json"
}

# Set the Streamlit page to run in wide mode by default
st.set_page_config(layout="wide")

# Paths to the video files
video_paths = ["videoplayback.mp4", "videoplayback2.mp4", "videoplayback3.mp4"]

st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    <h1 class="centered-title">Road Accident Detection</h1>
    """,
    unsafe_allow_html=True
)

# Initialize the model
model = AccidentDetectionModel("model copy.json", 'model_weights copy.keras')
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize the last message time
last_message_time = 0
last_log_time = 0

# Set title for the log in the sidebar
st.sidebar.title("Accident Log")

# Create placeholder for the log in the sidebar
log_placeholder = st.sidebar.empty()

# Initialize the log list
log_list = []

# CSV file path for logging
csv_file_path = "accident_logs.csv"

# Function to write log entry to CSV and update Streamlit sidebar
def log_to_csv_and_sidebar(current_time, camera_id, probability, notification_sent):
    date = time.strftime('%Y-%m-%d', time.localtime(current_time))
    time_entry = time.strftime('%H:%M:%S', time.localtime(current_time))
    
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([date, time_entry, camera_id, probability, 'Yes' if notification_sent else 'No'])

    # Update Streamlit sidebar
    log_entry = (
        f"Date: {date},<br>"
        f"Time: {time_entry},<br>"
        f"Camera_ID : {camera_id},<br>"
        f"Probability: {probability}%,<br>"
        f"Notification Sent: {'Yes' if notification_sent else 'No'}"
    )
    log_list.append(log_entry)
    log_placeholder.markdown("<br><br>".join(log_list), unsafe_allow_html=True)

async def send_notification():
    current_time = time.time()
    notificationapi.init(
        headers["clientId"],  # clientId
        headers["clientSecret"]  # clientSecret
    )

    await notificationapi.send({
        "notificationId": "ksp_datathon",
        "user": {
            "id": headers["email"],
            "number": headers["number"]  # Replace with your phone number
        },
        "mergeTags": {
            "comment": f"<br>Camera_ID : 16580 <br>Probability of an accident at Kothanur, Bengaluru, Karnataka 560077.<br>Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}<br><br>For the exact location, click here: <br>https://maps.app.goo.gl/YRGv6kR9SoTik5Sa7 ",
            "commentId": "testCommentId"
        }
    })

async def detect_accident(frame, camera_id):
    global last_message_time, last_log_time
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(gray_frame, (250, 250))

    pred, prob = model.predict_accident(roi[np.newaxis, :, :])
    if pred == "Accident":
        prob = round(prob[0][0] * 100, 2)
        cv2.rectangle(frame, (0, 0), (160, 20), (0, 0, 0), -1)
        cv2.putText(frame, pred + " " + str(prob), (10, 15), font, 0.5, (255, 255, 0), 2)
        current_time = time.time()
        notification_sent = False
        if prob > 99.50 and (time.time() - last_message_time) > 480:
            asyncio.create_task(send_notification())  # Create a task to send the notification
            last_message_time = time.time()
            notification_sent = True
            
        if prob > 95.50 and (time.time() - last_log_time) > 10:
            # Log the detection
            log_to_csv_and_sidebar(current_time, camera_id, prob, notification_sent)
            last_log_time = time.time()
    
    return frame  # Return the frame with detection results

async def stream_video(video_path, placeholder, width, height, camera_name, camera_id, detect=False):
    loop_count = 0
    while loop_count < 2:
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if detect:
                frame = await detect_accident(frame, camera_id)  # Await the detection function
            
            # Add the camera ID text on the frame
            cv2.putText(frame, f"Camera_ID : {camera_id}", (20, frame.shape[0] - 20), font, 0.5, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            encoded_frame = base64.b64encode(frame_bytes).decode()
            video_str = f'''
                <h4>{camera_name}</h4>
                <img width="{width}" height="{height}" src="data:image/jpeg;base64,{encoded_frame}">
            '''
            placeholder.markdown(video_str, unsafe_allow_html=True)
            await asyncio.sleep(0.0003)  # Control the frame rate
        loop_count += 1

async def main():
    # Create columns for each camera
    col1, col2, col3 = st.columns(3)
    
    # Create placeholders for each camera
    placeholders = [col1.empty(), col2.empty(), col3.empty()]
    
    # Run the streams concurrently
    await asyncio.gather(
        stream_video(video_paths[0], placeholders[0], 480, 320, "Camera 1", 1650, detect=True),
        stream_video(video_paths[1], placeholders[1], 480, 320, "Camera 2", 1990, detect=False),
        stream_video(video_paths[2], placeholders[2], 480, 320, "Camera 3", 89650, detect=False)
    )

# Initialize a ThreadPoolExecutor
executor = ThreadPoolExecutor()

# Run the async main function
asyncio.run(main())

# Shut down the ThreadPoolExecutor
executor.shutdown(wait=True)
