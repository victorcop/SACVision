from ultralytics import YOLO
from roboflow import Roboflow

def create_model():
    rf = Roboflow(api_key="XXXXX")
    project = rf.workspace("sacvision").project("sac_8xx_dataset")
    version = project.version(7)
    dataset = version.download("yolov11")

