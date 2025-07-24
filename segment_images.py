from ultralytics import YOLO
import os

model = YOLO('yolov8n-seg.pt')  

input_folder = 'images1'
output_dir = 'segmented_outputs1'

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, file)
        print(f"Segmenting: {file}")
        model.predict(source=image_path, save=True, project=output_dir, name='results')
