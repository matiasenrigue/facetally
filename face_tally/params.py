import os

##################  CONSTANTS  ##################
LOCAL_DATA_PATH = os.path.expanduser("~/.lewagon/facetally_data")
class_mapping = {0: "face"}

##################  VARIABLES  ##################
SPLIT_RATIO = float(os.environ.get("SPLIT_RATIO"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE"))
EPOCH = int(os.environ.get("EPOCH"))
GLOBAL_CLIPNORM = float(os.environ.get("GLOBAL_CLIPNORM"))
BOX_FORMAT = os.environ.get("BOX_FORMAT")
BACKBONE_SIZE = os.environ.get("BACKBONE_SIZE")
MODEL_SOURCE = os.environ.get("MODEL_SOURCE")

# Google Cloud Storage
GCP_REGION = os.environ.get("GCP_REGION")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BUCKET_NAME = os.environ.get("BUCKET_NAME")

# PREFECT
PREFECT_BLOCK = os.environ.get("PREFECT_BLOCK")

# COMET
COMET_API_KEY = os.environ["COMET_API_KEY"]
COMET_PROJECT_NAME = os.environ["COMET_PROJECT_NAME"]
COMET_MODEL_NAME = os.environ["COMET_MODEL_NAME"]
COMET_WORKSPACE_NAME = os.environ["COMET_WORKSPACE_NAME"]
