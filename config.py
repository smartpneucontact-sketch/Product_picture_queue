import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///queue.db')
    # Fix for Railway PostgreSQL URLs
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Cloudinary
    CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME')
    CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY')
    CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET')
    
    # remove.bg API
    REMOVEBG_API_KEY = os.environ.get('REMOVEBG_API_KEY')
    
    # Shopify
    SHOPIFY_STORE_URL = os.environ.get('SHOPIFY_STORE_URL')
    SHOPIFY_ACCESS_TOKEN = os.environ.get('SHOPIFY_ACCESS_TOKEN')
    
    # Upload secret (shared with Raspberry Pi)
    UPLOAD_SECRET = os.environ.get('UPLOAD_SECRET', 'change-me-in-production')
    
    # Processing settings
    OUTPUT_IMAGE_SIZE = 1000
