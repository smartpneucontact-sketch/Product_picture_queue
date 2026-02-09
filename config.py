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
    
    # Cloudflare R2 Storage
    R2_ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID')
    R2_SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY')
    R2_BUCKET_NAME = os.environ.get('R2_BUCKET_NAME', 'smartpneu-images')
    R2_ACCOUNT_ID = os.environ.get('R2_ACCOUNT_ID', 'e276b85cfe3f10140f3eaa752e405d39')
    R2_PUBLIC_URL = os.environ.get('R2_PUBLIC_URL', 'https://pub-7541f1c3a1ac4cd1a1af891329a72c80.r2.dev')
    
    # remove.bg API (legacy)
    REMOVEBG_API_KEY = os.environ.get('REMOVEBG_API_KEY')
    
    # Poof API (removebgapi.com) - recommended
    POOF_API_KEY = os.environ.get('POOF_API_KEY')
    
    # Shopify
    SHOPIFY_STORE_URL = os.environ.get('SHOPIFY_STORE_URL')
    SHOPIFY_ACCESS_TOKEN = os.environ.get('SHOPIFY_ACCESS_TOKEN')
    
    # Upload secret (shared with Raspberry Pi)
    UPLOAD_SECRET = os.environ.get('UPLOAD_SECRET', 'change-me-in-production')
    
    # Processing settings
    OUTPUT_IMAGE_SIZE = int(os.environ.get('OUTPUT_IMAGE_SIZE', 2048))  # Shopify recommends 2048x2048
