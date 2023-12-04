from prefect_gcp import GcpCredentials
from google.cloud import storage
from face_tally.params import *


async def load_google_credentials():
    gcp_credentials = await GcpCredentials.load(PREFECT_BLOCK)
    return gcp_credentials.get_credentials_from_service_account()


async def create_google_cloud_client():
    credentials = await load_google_credentials()
    client = storage.client(credentials=credentials)
    return client
