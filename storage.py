import cloudinary
import cloudinary.uploader
from datetime import datetime
import uuid

class CloudinaryStorage:
    def __init__(self, cloud_name, api_key, api_secret):
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True
        )
    
    def upload_image(self, file_data, filename, folder='originals'):
        """Upload an image to Cloudinary and return the URL."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        public_id = f"smartpneu/{folder}/{timestamp}_{unique_id}"
        
        result = cloudinary.uploader.upload(
            file_data,
            public_id=public_id,
            resource_type="image"
        )
        
        return result['secure_url']
    
    def upload_processed_image(self, file_data, original_filename):
        """Upload a processed image to the 'processed' folder."""
        return self.upload_image(file_data, original_filename, folder='processed')
    
    def delete_image(self, url):
        """Delete an image from Cloudinary by its URL."""
        try:
            # Extract public_id from URL
            # URL format: https://res.cloudinary.com/cloud_name/image/upload/v123/public_id.jpg
            parts = url.split('/upload/')
            if len(parts) > 1:
                public_id = parts[1].rsplit('.', 1)[0]  # Remove extension
                # Remove version if present (v123456/)
                if public_id.startswith('v') and '/' in public_id:
                    public_id = public_id.split('/', 1)[1]
                cloudinary.uploader.destroy(public_id)
            return True
        except Exception as e:
            print(f"Error deleting image: {e}")
            return False