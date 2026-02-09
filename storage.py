import boto3
from botocore.config import Config as BotoConfig
from datetime import datetime
import uuid

class R2Storage:
    def __init__(self, access_key_id, secret_access_key, bucket_name, account_id, public_url):
        self.bucket_name = bucket_name
        self.public_url = public_url.rstrip('/')
        
        # R2 endpoint URL
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=BotoConfig(signature_version='s3v4'),
            region_name='auto'
        )
    
    def upload_image(self, file_data, filename, folder='originals'):
        """Upload an image to R2 and return the public URL."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        
        # Determine extension
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'jpg'
        
        # Create object key
        object_key = f"{folder}/{timestamp}_{unique_id}.{ext}"
        
        # Determine content type
        content_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        content_type = content_types.get(ext, 'image/jpeg')
        
        # Upload to R2
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=object_key,
            Body=file_data,
            ContentType=content_type
        )
        
        # Return public URL
        return f"{self.public_url}/{object_key}"
    
    def upload_processed_image(self, file_data, original_filename):
        """Upload a processed image to the 'processed' folder."""
        return self.upload_image(file_data, original_filename, folder='processed')
    
    def delete_image(self, url):
        """Delete an image from R2 by its URL."""
        try:
            # Extract object key from URL
            # URL format: https://pub-xxx.r2.dev/folder/filename.jpg
            if self.public_url in url:
                object_key = url.replace(self.public_url + '/', '')
                self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=object_key
                )
            return True
        except Exception as e:
            print(f"Error deleting image: {e}")
            return False
