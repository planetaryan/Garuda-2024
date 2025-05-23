from ultralytics import YOLO

# Load a model
model = YOLO('./best.pt')  # load a custom trained model

# Export the model
model.export(format='engine', half=True, int8=True, simplify=True)