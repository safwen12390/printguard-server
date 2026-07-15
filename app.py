import base64, os, time, uuid
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch
import torch.nn.modules.container  # <-- ADDED THIS IMPORT
import ultralytics.nn.tasks

# ADD BOTH SAFE GLOBALS BEFORE LOADING MODEL
torch.serialization.add_safe_globals([
    torch.nn.modules.container.Sequential,  # <-- ADDED THIS
    ultralytics.nn.tasks.DetectionModel
])

# Download model from Hugging Face if not exists
if not os.path.exists("best.pt"):
    print("📥 Downloading model from HuggingFace...")
    hf_hub_download(
        repo_id="safwennnnn/printguard-model",
        filename="best.pt",
        local_dir=".",
        token=os.environ.get("HF_TOKEN")
    )
    print("✅ Model downloaded!")

app = Flask(__name__)
CORS(app)

model = YOLO("best.pt")
print(f"✅ Model loaded. Classes: {model.names}")

HISTORY = []

def run_inference(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return [], None
    results = model(frame, conf=0.25, verbose=False)
    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 4),
            "x1": round(float(box.xyxy[0][0]), 1),
            "y1": round(float(box.xyxy[0][1]), 1),
            "x2": round(float(box.xyxy[0][2]), 1),
            "y2": round(float(box.xyxy[0][3]), 1),
        })
    annotated = results[0].plot()
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    return detections, b64

@app.route("/status", methods=["GET"])
def status():
    cap = cv2.VideoCapture(0)
    cam_ok = cap.isOpened()
    cap.release()
    last = HISTORY[-1]["timestamp"] if HISTORY else None
    return jsonify({
        "model_loaded": True,
        "camera_connected": cam_ok,
        "total_detections": len(HISTORY),
        "last_detection": last,
        "version": "1.0.0",
    })

@app.route("/upload_raw", methods=["POST"])
def upload_raw():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    image_bytes = request.files["image"].read()
    detections, b64 = run_inference(image_bytes)
    HISTORY.append({"id": str(uuid.uuid4()), "timestamp": datetime.utcnow().isoformat(), "detections": detections})
    if len(HISTORY) > 100: HISTORY.pop(0)
    return jsonify({"detections": detections, "annotated_image": b64})

@app.route("/detect", methods=["GET"])
def detect():
    if not HISTORY: return jsonify({"detections": [], "annotated_image": None})
    return jsonify(HISTORY[-1])

@app.route("/stream", methods=["GET"])
def stream():
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            err = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(err, "No camera", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            _, buf = cv2.imencode(".jpg", err)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            return
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                results = model(frame, conf=0.25, verbose=False)
                annotated = results[0].plot()
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                time.sleep(0.1)
        finally:
            cap.release()
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/alert", methods=["POST"])
def alert():
    data = request.get_json(silent=True) or {}
    return jsonify({"ok": True, "detection_id": data.get("detection_id")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
