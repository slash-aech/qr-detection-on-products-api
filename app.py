from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO
import uuid
import os
import shutil
import cv2

app = FastAPI()
model = YOLO("my_model/my_model.pt")
class_names = model.names

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    input_filename = f"input_{uuid.uuid4().hex}.jpg"
    with open(input_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run inference
    results = model(input_filename)
    img = cv2.imread(input_filename)

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < 0.5:
            continue

        cls_id = int(box.cls[0])
        label = class_names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Blur only if it's warranty_qr(hide)
        #if label == "warranty_qr(hide)":
           # roi = img[y1:y2, x1:x2]
            #blurred = cv2.GaussianBlur(roi, (45, 45), 0)
            #img[y1:y2, x1:x2] = blurred

        # Draw box (after blur if any)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        label_text = f"{label} {conf:.2f}"
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # Save final image
    output_filename = f"output_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(output_filename, img)

    return FileResponse(output_filename, media_type="image/jpeg")