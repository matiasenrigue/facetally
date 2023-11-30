import os

##################  VARIABLES  ##################
SPLIT_RATIO = float(os.environ.get("SPLIT_RATIO"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE"))
EPOCH = int(os.environ.get("EPOCH"))

GLOBAL_CLIPNORM = float(os.environ.get("GLOBAL_CLIPNORM"))

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")

BOX_FORMAT = os.environ.get("BOX_FORMAT")
LOCAL_DATA_PATH = os.path.expanduser("~/.lewagon/facetally_data")
class_mapping = {0: "face"}
