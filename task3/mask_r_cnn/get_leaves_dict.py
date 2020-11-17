import os
import glob
from detectron2.structures import BoxMode

def get_leave_dict(dataset):
    image_path = glob.glob(os.path.join(dataset, '*_'))
