from flask import Flask, request, send_file
from ultralytics import YOLO
import tempfile
import cv2
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Lazy model loader
model = None
def get_model():
    global model
    if model is None:
        model = YOLO("my_model/my_model.pt")
    return model

@app.route("/predict/", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files['file']

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jpg")
            output_path = os.path.join(tmpdir, "output.jpg")
            file.save(input_path)

            model = get_model()
            results = model(input_path)
            img = cv2.imread(input_path)
            class_names = model.names

            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                cls_id = int(box.cls[0])
                label = class_names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # blur logic
                if label == "warranty_qr(hide)":
                    roi = img[y1:y2, x1:x2]
                    blurred = cv2.GaussianBlur(roi, (45, 45), 0)
                    img[y1:y2, x1:x2] = blurred

                # bounding box color
                box_color = (0, 0, 255) if label == "warranty_qr(hide)" else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=box_color, thickness=2)

                # label text
                label_text = f"{label} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 4, y1), box_color, -1)
                cv2.putText(img, label_text, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

            cv2.imwrite(output_path, img)
            return send_file(output_path, mimetype="image/jpeg")

    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/health")
def health():
    return {"status": "ok"}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
