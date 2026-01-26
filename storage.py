from google.cloud import storage
from google.oauth2 import service_account
import json
import os
from datetime import datetime
import uuid

class GCSStorage:
    def __init__(self, bucket_name, credentials_json=None):
        self.bucket_name = bucket_name
        
        if credentials_json:
            # Parse credentials from JSON string (for Railway env var)
            credentials_dict = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            self.client = storage.Client(credentials=credentials, project=credentials_dict.get('project_id'))
        else:
            # Use default credentials (for local dev with gcloud auth)
            self.client = storage.Client()
        
        self.bucket = self.client.bucket(bucket_name)
    
    def upload_image(self, file_data, filename, folder='originals'):
        """Upload an image to GCS and return the public URL."""
        # Generate unique filename to avoid collisions
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        ext = os.path.splitext(filename)[1] or '.jpg'
        blob_name = f"{folder}/{timestamp}_{unique_id}{ext}"
        
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(file_data, content_type='image/jpeg')
        
        # Make publicly readable
        blob.make_public()
        
        return blob.public_url
    
    def upload_processed_image(self, file_data, original_filename):
        """Upload a processed image to the 'processed' folder."""
        return self.upload_image(file_data, original_filename, folder='processed')
    
    def delete_image(self, url):
        """Delete an image from GCS by its URL."""
        try:
            # Extract blob name from URL
            # URL format: https://storage.googleapis.com/bucket-name/blob-name
            blob_name = url.split(f'{self.bucket_name}/')[-1]
            blob = self.bucket.blob(blob_name)
            blob.delete()
            return True
        except Exception as e:
            print(f"Error deleting image: {e}")
            return False
