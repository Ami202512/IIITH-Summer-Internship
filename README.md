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

### Setup

- **Model Used:** `yolov8l-seg.pt`
- **Tools:** Python, FFmpeg, Ultralytics YOLOv8
- **Dependencies:**
  ```bash
  pip install ultralytics
  sudo apt install ffmpeg
  ```

---

### Part 1: Segmentation from Video

#### 1. Extract Frames from Video

Downloaded a sample road video from [Pexels](https://www.pexels.com/video/dash-cam-view-of-the-road-5921059/) and placed it in the root folder as `road.mp4`.

Used FFmpeg to extract frames at 2 FPS:

```bash
ffmpeg -i road.mp4 -vf "fps=2" images/roadframe_%03d.jpg
```

#### 2. Run YOLOv8 Segmentation

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

#### 3. Rename for Video Stitching

Inside each result folder (e.g. `video_input/seg_results/`), run:

```powershell
$i=1; Get-ChildItem *.jpg | Sort-Object Name | ForEach-Object {Rename-Item $_ -NewName ("{0:D4}.jpg" -f $i); $i++}
```

#### 4. Convert Back to Video

Use FFmpeg to combine frames back into a video:

```bash
ffmpeg -framerate 5 -i %04d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4
```

---


### Outcome

- Successfully segmented frames using YOLOv8.
- Generated videos of segmented frames from both dynamic (video) and static (image) sources.
- Demonstrated automated image-to-video pipeline using FFmpeg + YOLOv8.


## Task 3: Object Detection on African Wildlife Dataset Using YOLOv8n

This task involved training the YOLOv8n model using the African Wildlife object detection dataset to detect and classify animals in images. The dataset was split into train, validation, and test sets, and the model was trained on Google Colab with GPU.

---

### Download Dataset

The African Wildlife dataset is required for training and evaluation but is not included in this repository due to size constraints.

**To use this project:**

1. Download the dataset from [(https://docs.ultralytics.com/datasets/detect/african-wildlife/#dataset-yaml)].
2. After downloading, unzip it and place the contents into the `datasets/african-wildlife/` directory as shown above.


The dataset is structured in the YOLO format:

```
datasets/african-wildlife/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

### Training Details

- **Model**: YOLOv8n
- **Environment**: Google Colab (GPU)
- **Framework**: Ultralytics YOLOv8
- **Command Used**:

```bash
!yolo detect train \
  model=yolov8n.pt \
  data=african_wildlife.yaml \
  epochs=20 \
  imgsz=640 \
  batch=16 \
  project=runs/detect \
  name=train3
```

---

### Evaluation Metrics

| Metric           | Value |
|------------------|-------|
| Precision        | 0.914 |
| Recall           | 0.895 |
| mAP@0.5          | 0.942 |
| mAP@0.5:0.95     | 0.780 |

---

### Interpretation of Results

1. **Loss curves** in `results.png` show a steady decrease, indicating successful learning.
2. **Confusion matrix** reveals strong diagonal dominance, indicating high classification accuracy across all classes.
3. **Precision and Recall curves** show balanced and high performance.
4. **F1 Curve** highlights stable improvement across epochs.
5. **Class-level observation**:
   - *Rhino* achieved the highest accuracy.
   - *Elephant* had slightly lower metrics, likely due to fewer examples or more variability.

---

### Visual Results

All result visualizations are in the `images/` folder:

- `predictions.png`: Correct detections and classifications on sample images
- `confmat.png`
- `F1.png`
- `RC.png`
- `RC.png`
- `PR.png`
- `results.png` (e.g., `train/box_loss`, `metrics/mAP_0.5`, etc.)

---

### Output Directory Structure

After training, YOLOv8 saves results to:

```
runs/detect/train3/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.png
├── confmat.png
├── PR.png
├── F1.png
├── training logs and metrics
```

---

### Conclusion

The YOLOv8n model trained on the African Wildlife dataset performed very well, achieving high precision and recall. It generalized effectively and produced clean, interpretable results. This task successfully demonstrates object detection with a lightweight model.

---

## Task 4: Custom Dataset Creation and YOLOv8 Training

**Objective:**  
To create a custom object detection dataset from a YouTube video, annotate it, train a YOLOv8 model on it, and test predictions.

### Steps Involved:

1. **Video to Image Conversion:**  
   A YouTube video of an aquarium scene was downloaded and split into frames using `yt-dlp` and `ffmpeg`.

2. **Dataset Creation & Annotation:**  
   - Selected frames were uploaded to [Roboflow](https://roboflow.com/).
   - Annotated manually using bounding boxes for:
     - `fish`
     - `starfish`
     - `stingray`
   - Exported the dataset in YOLOv8 format with `train`, `valid`, and `test` splits.

3. **Training the Model:**  
   - YOLOv8s model was used for training.
   - Custom `underwater3.yaml` config was created on-the-fly inside the training script.
   - Trained for **30 epochs** on the dataset with image size `640`.

4. **Testing and Predictions:**  
   - The best model checkpoint was used to predict on test images.
   - Predictions were saved and a few were visualized.

---

### Sample Predictions

Here are some results from the model:

<p align="center">
  <img src="Task 4/predictions/1.jpg" width="30%"/>
  <img src="Task 4/predictions/2.jpg" width="30%"/>
  <img src="Task 4/predictions/3.jpg" width="30%"/>
</p>

---

### 📈 Training Results

<img src="Task 4/results.png" width="80%"/>

---

### Code

The full pipeline from training to prediction is available in  
`Task 4/marine.py`.

