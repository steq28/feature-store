import hsfs
from config.constants import HOST_INSTANCE, PROJECT_NAME, API_KEY

def connect_to_hopsworks():
    conn = hsfs.connection(
        host = HOST_INSTANCE,             # Port to reach your Hopsworks instance, defaults to 443
        project=PROJECT_NAME,               # Name of your Hopsworks Feature Store project
        api_key_value = API_KEY,             # The API key to authenticate with Hopsworks
        hostname_verification=True          # Disable for self-signed certificates
    )

    fs = conn.get_feature_store() 
    return fs