import os
from ultralytics import YOLO

MODEL_PATH = "runs/segment/train/weights/best.pt" 
IMAGE_DIR = "path to data file"

def detect_cracks():
    if not os.path.isdir(IMAGE_DIR):
        return

    model = YOLO(MODEL_PATH)
    
    model.predict(
        source=IMAGE_DIR, 
        save=True,
        show=False,
        conf=0.25,
        project='crack_detection_output',
        name='crack_prediction',
        boxes=False,
        show_labels=False,
        show_conf=False
    )

if __name__ == "__main__":
    detect_cracks()