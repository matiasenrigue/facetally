import os

##################  VARIABLES  ##################
SPLIT_RATIO = float(os.environ.get("SPLIT_RATIO"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE"))
EPOCH = int(os.environ.get("EPOCH"))
GLOBAL_CLIPNORM = float(os.environ.get("GLOBAL_CLIPNORM"))
BOX_FORMAT = os.environ.get("BOX_FORMAT")
LOCAL_DATA_PATH = os.path.expanduser("~/.lewagon/project_data")

class_mapping = {0: "face"}
