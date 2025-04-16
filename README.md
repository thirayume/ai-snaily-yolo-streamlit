# AI Snaily - YOLO Detection App

A Streamlit application that allows users to run object detection using multiple YOLO model versions (v5, v8, v10, v11).

## Features

- Upload multiple images (JPG, PNG, GIF)
- Select which YOLO model versions to use
- View detection progress in real-time
- Compare detection results across different YOLO versions
- Preview images in lightbox mode
- Download detection results

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your YOLO model files in the `models` directory:
   - v5.pt (download from [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5))
   - v8.pt (download from [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics))
   - v10.pt (download from [THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10))
   - v11.pt (download from [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics))