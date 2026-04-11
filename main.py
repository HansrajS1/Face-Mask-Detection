import cv2
import numpy as np
import asyncio
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from tensorflow.keras.models import load_model

app = FastAPI()

# ===============================
# CORS (RENDER SAFE)
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# MODEL
# ===============================
model = load_model("fixed_model.keras", compile=False)
IMG_SIZE = (224, 224)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cv2.setNumThreads(1)

# ===============================
# GLOBAL FRAME (NO CAMERA ON SERVER)
# ===============================
latest_frame = None
latest_result = []
lock = threading.Lock()

# ===============================
# AI PROCESSOR THREAD
# ===============================
def process_frames():
    global latest_frame, latest_result

    while True:
        if latest_frame is None:
            continue

        with lock:
            frame = latest_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)

        results = []

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face = cv2.resize(face, IMG_SIZE)
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            pred = model.predict(face, verbose=0)[0]
            score = float(pred[0])

            label = "With Mask" if score <= 0.4 else "Without Mask"
            color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)

            results.append((x, y, w, h, label, color))

        with lock:
            latest_result = results

threading.Thread(target=process_frames, daemon=True).start()

# ===============================
# WEBSOCKET (LIVE STREAM)
# ===============================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global latest_frame, latest_result

    await websocket.accept()
    print("Client connected")

    try:
        while True:
            data = await websocket.receive_bytes()

            frame = cv2.imdecode(
                np.frombuffer(data, np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                continue

            frame = cv2.resize(frame, (320, 240))

            with lock:
                latest_frame = frame.copy()
                results = latest_result.copy()

            output = frame.copy()

            for (x, y, w, h, label, color) in results:
                cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
                cv2.putText(output, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            _, buffer = cv2.imencode(".jpg", output)

            await websocket.send_bytes(buffer.tobytes())

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print("Client disconnected")

# ===============================
# 📸 SNAPSHOT (NO CAMERA)
# ===============================
@app.get("/snapshot")
def snapshot():
    global latest_frame

    if latest_frame is None:
        return {"error": "No frame received yet from WebSocket"}

    with lock:
        frame = latest_frame.copy()
        results = latest_result.copy()

    for (x, y, w, h, label, color) in results:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _, buffer = cv2.imencode(".jpg", frame)

    return StreamingResponse(iter([buffer.tobytes()]), media_type="image/jpeg")

# ===============================
# 📸 IMAGE UPLOAD
# ===============================
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    img = cv2.resize(img, (320, 240))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]

        face = cv2.resize(face, IMG_SIZE)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face, verbose=0)[0]
        score = float(pred[0])

        label = "With Mask" if score <= 0.4 else "Without Mask"
        color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _, buffer = cv2.imencode(".jpg", img)

    return StreamingResponse(iter([buffer.tobytes()]), media_type="image/jpeg")

# ===============================
# 🎬 VIDEO UPLOAD
# ===============================
@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    contents = await file.read()

    path = "temp.mp4"
    with open(path, "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture(path)

    def generate():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (320, 240))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 4)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]

                face = cv2.resize(face, IMG_SIZE)
                face = face.astype("float32") / 255.0
                face = np.expand_dims(face, axis=0)

                pred = model.predict(face, verbose=0)[0]
                score = float(pred[0])

                label = "With Mask" if score <= 0.4 else "Without Mask"
                color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            _, buffer = cv2.imencode(".jpg", frame)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ===============================
# HOME
# ===============================
@app.get("/")
def home():
    return {
        "status": "RUNNING ON RENDER",
        "message": "Face Mask Detection API",
        "endpoints": {
            "ws": "/ws",
            "snapshot": "/snapshot",
            "upload_image": "/upload-image",
            "upload_video": "/upload-video"
        }
    }