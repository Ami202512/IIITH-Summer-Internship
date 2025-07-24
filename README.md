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
```


## Task 2: YOLOv8 Segmentation on Video and Static Images

This task involves applying YOLOv8 segmentation on both a video input and a set of static images. Below are the steps and methods I used.

---

## Setup

- **Model Used:** `yolov8l-seg.pt`
- **Tools:** Python, FFmpeg, Ultralytics YOLOv8
- **Dependencies:**
  ```bash
  pip install ultralytics
  sudo apt install ffmpeg
  ```

---

## Part 1: Segmentation from Video

### 1. Extract Frames from Video

Downloaded a sample road video from [Pexels](https://www.pexels.com/video/dash-cam-view-of-the-road-5921059/) and placed it in the root folder as `road.mp4`.

Used FFmpeg to extract frames at 2 FPS:

```bash
ffmpeg -i road.mp4 -vf "fps=2" images/roadframe_%03d.jpg
```

### 2. Run YOLOv8 Segmentation

```python
# segment_images.py
from ultralytics import YOLO
import os

model = YOLO("yolov8l-seg.pt")

folders = ["images", "images1"]
output_folders = ["video_input", "video_input1"]

for image_folder, output_name in zip(folders, output_folders):
    output_folder = os.path.join("segmented_outputs", output_name)
    os.makedirs(output_folder, exist_ok=True)

    for img in os.listdir(image_folder):
        if img.endswith(".jpg"):
            model.predict(
                source=os.path.join(image_folder, img),
                save=True,
                project=output_folder,
                name="seg_results",
                save_txt=False
            )
```

- `images/` contains frames extracted from the video.
- `images1/` contains static road images from Google.

### 3. Rename for Video Stitching

Inside each result folder (e.g. `video_input/seg_results/`), run:

```powershell
$i=1; Get-ChildItem *.jpg | Sort-Object Name | ForEach-Object {Rename-Item $_ -NewName ("{0:D4}.jpg" -f $i); $i++}
```

### 4. Convert Back to Video

Use FFmpeg to combine frames back into a video:

```bash
ffmpeg -framerate 5 -i %04d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4
```

---


## Outcome

- Successfully segmented frames using YOLOv8.
- Generated videos of segmented frames from both dynamic (video) and static (image) sources.
- Demonstrated automated image-to-video pipeline using FFmpeg + YOLOv8.
