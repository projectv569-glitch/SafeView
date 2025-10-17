import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog

# -----------------------------
# Load trained model
# -----------------------------
model_path = "violence_detection_model.h5"
model = load_model(model_path)
input_height, input_width = model.input_shape[1], model.input_shape[2]
print(f"Model loaded. Expected input size: {input_width}x{input_height}")

# -----------------------------
# Preprocess frame
# -----------------------------


def preprocess_frame(frame):
    frame = cv2.resize(frame, (input_width, input_height))
    frame = frame.astype('float32') / 255.0
    return np.expand_dims(frame, axis=0)


def extract_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (input_width, input_height))
        frame_norm = frame_resized.astype('float32') / 255.0
        frames.append(frame_norm)
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()
    return np.array(frames)

# -----------------------------
# Classify and process video
# -----------------------------


def classify_video(video_path):
    frames_array = extract_frames(video_path)
    if len(frames_array) == 0:
        print("❌ No frames extracted from video.")
        return

    # Predict violence on all frames
    preds = model.predict(frames_array, verbose=0)
    avg_pred = np.mean(preds)
    is_violent = avg_pred > 0.7  # threshold (adjustable)

    # Open video again for processing
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(os.path.dirname(video_path),
                            "output_violence_check.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(
        f"Processing video... {'⚠️ Violent' if is_violent else '✅ Non-violent'}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if is_violent:
            overlay_color = np.full_like(frame, (0, 0, 255))  # Hard red shield
            frame = cv2.addWeighted(overlay_color, 0.9, frame, 0.1, 0)

        out.write(frame)

        # Show live
        cv2.imshow("Processed Video", frame)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Result
    if is_violent:
        print(f"⚠️ Video is VIOLENT! Violence score: {avg_pred:.2f}")
    else:
        print(f"✅ Video is non-violent. Violence score: {avg_pred:.2f}")

    print(f"Processed video saved at: {out_path}")
    os.startfile(out_path)  # Automatically open the processed video

# -----------------------------
# Video upload
# -----------------------------


def upload_video():
    root = Tk()
    root.withdraw()  # hide main window
    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )
    if file_path:
        classify_video(file_path)
    else:
        print("No file selected.")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    upload_video()
