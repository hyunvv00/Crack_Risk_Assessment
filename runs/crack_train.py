from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="crack-seg.yaml", epochs=100, imgsz=640, batch=16, workers=64)
