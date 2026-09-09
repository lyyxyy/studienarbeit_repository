"""Download Carla_Labeling v5 dataset from Roboflow (yolov11 format)."""
import os
from roboflow import Roboflow

# Download into this directory, mirroring the author's original layout
TARGET_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TARGET_DIR)

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("surajworkplace").project("carla_labeling")
version = project.version(5)
dataset = version.download("yolov11")
print("DOWNLOAD COMPLETE:", dataset.location)
