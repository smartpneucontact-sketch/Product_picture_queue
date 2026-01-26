import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///queue.db')
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Google Cloud Storage
    GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')
    GCS_CREDENTIALS_JSON = os.environ.get('GCS_CREDENTIALS_JSON')  # JSON string of service account
    
    # remove.bg API
    REMOVEBG_API_KEY = os.environ.get('REMOVEBG_API_KEY')
    
    # Shopify
    SHOPIFY_STORE_URL = os.environ.get('SHOPIFY_STORE_URL')  # e.g., smartpneu.myshopify.com
    SHOPIFY_ACCESS_TOKEN = os.environ.get('SHOPIFY_ACCESS_TOKEN')
    
    # Upload secret (simple auth for Pi uploads)
    UPLOAD_SECRET = os.environ.get('UPLOAD_SECRET', 'change-me-in-production')
    
    # Processing settings
    OUTPUT_IMAGE_SIZE = 1000  # 1000x1000 pixels
