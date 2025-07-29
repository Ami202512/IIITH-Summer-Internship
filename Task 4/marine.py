# Install required libraries
# (Only for Colab; you can remove these lines locally if not needed)
# !pip install -q ultralytics yt-dlp

# Create dataset config YAML in code
yaml = """
train: /content/underwater_data/aquarium_pretrain/train/images
val: /content/underwater_data/aquarium_pretrain/valid/images
test: /content/underwater_data/aquarium_pretrain/test/images

nc: 3
names: ['fish', 'starfish', 'stingray']
"""

with open("underwater3.yaml", "w") as f:
    f.write(yaml)

# Train the YOLOv8 model
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(data="underwater3.yaml", epochs=30, imgsz=640)

# Load best trained model and predict
model = YOLO('runs/detect/train3/weights/best.pt')
model.predict(source="/content/underwater_data/aquarium_pretrain/test/images", save=True, imgsz=640)

# Display first few predicted images (Colab)
import os
from IPython.display import Image, display
predict_folder = "runs/detect/predict"
for file_name in os.listdir(predict_folder)[:3]:
    if file_name.endswith(".jpg"):
        display(Image(filename=os.path.join(predict_folder, file_name)))

# Show training result graph
from IPython.display import Image
display(Image(filename='runs/detect/train2/results.png'))
