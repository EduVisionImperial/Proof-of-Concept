from ultralytics import YOLO

model = YOLO('yolo11m.pt')
model.train(data='custom_db.yaml', epochs=25, batch=16, workers=1, imgsz=640, device=0)