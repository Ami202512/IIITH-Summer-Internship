## Task 1: YOLOv8 Object Detection & Segmentation (Local - VS Code)

Performed object detection and image segmentation using Ultralytics YOLOv8 models on Windows 11 using VS Code terminal.

### Files
- `bus.jpg` – input image
- `yolov8n.pt` – object detection model
- `yolov8l-seg.pt` – segmentation model
- Outputs:
  - `runs/yolov8n_detect/bus.jpg` → detection result
  - `runs/segment/predict/bus.jpg` → segmentation result

### Steps Run in Terminal

```bash
# Create virtual environment and activate
python -m venv venv
venv\Scripts\activate

# Install YOLOv8
pip install ultralytics

# Run detection
yolo predict model=yolov8n.pt source=bus.jpg project=runs name=yolov8n_detect

# Run segmentation
yolo predict model=yolov8l-seg.pt source=bus.jpg project=runs name="segment/predict"
