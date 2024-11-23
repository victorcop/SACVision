from ultralytics import YOLO
from roboflow import Roboflow

def create_model():
    rf = Roboflow(api_key="2il7oLFEnQHkVDoa5323")
    project = rf.workspace("sacvision").project("sac_8xx_dataset")
    version = project.version(7)
    dataset = version.download("yolov11")

