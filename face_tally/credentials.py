from prefect_gcp import GcpCredentials
from google.cloud import storage
from face_tally.params import *


async def create_google_cloud_client():
    gcp_credentials = await GcpCredentials.load(PREFECT_BLOCK)
    return gcp_credentials.get_cloud_storage_client()
