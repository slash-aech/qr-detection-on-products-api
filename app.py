from flask import Flask, request, send_file
from ultralytics import YOLO
import uuid
import os
import cv2
from flask_cors import CORS

app = Flask(__name__)

#CORS setup
CORS(app)

#yolo model
model = YOLO("my_model/my_model.pt")
class_names = model.names

@app.route("/predict/", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files['file']
    input_filename = f"input_{uuid.uuid4().hex}.jpg"
    file.save(input_filename)

    results = model(input_filename)
    img = cv2.imread(input_filename)

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < 0.5:
            continue

        cls_id = int(box.cls[0])
        label = class_names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        #blur logic
        if label == "warranty_qr(hide)":
            roi = img[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(roi, (45, 45), 0)
            img[y1:y2, x1:x2] = blurred

        #bound box color assign
        if label == "warranty_qr(hide)":
            box_color = (0, 0, 255)
        else:
            box_color = (0, 255, 0)

        #draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color=box_color, thickness=2)

        #label text
        label_text = f"{label} {conf:.2f}"
        font_scale = 0.6
        font_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        # label text background
        text_x1, text_y1 = x1, y1 - text_height - 10
        text_x2, text_y2 = x1 + text_width + 4, y1
        cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), box_color, -1)

        #white text
        cv2.putText(img, label_text, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

    #save final image
    output_filename = f"output_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(output_filename, img)

    return send_file(output_filename, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)
