# QR Detection API

This project provides an API that scans uploaded images for QR codes—especially those that may contain sensitive information like warranty details—and automatically blurs them using a custom-trained YOLO model.

## Endpoint

`POST /predict/`

**Request:**  
- Multipart image file (JPEG or PNG)

**Response:**  
- Processed image (JPEG) with bounding boxes

## How to Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
````

2. Start the server:

```bash
python app.py
```

3. Open the API docs at:

```
http://127.0.0.1:5000/predict/
```
