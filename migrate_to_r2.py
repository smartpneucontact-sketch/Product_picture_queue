#!/usr/bin/env python3
"""
Migration script: Copy all Cloudinary images to R2 and update database URLs.
Run this once to migrate existing images before Cloudinary account is suspended.

Usage:
    python migrate_to_r2.py

Environment variables required:
    - DATABASE_URL
    - R2_ACCESS_KEY_ID
    - R2_SECRET_ACCESS_KEY
    - R2_BUCKET_NAME (optional, defaults to 'smartpneu-images')
    - R2_ACCOUNT_ID (optional)
    - R2_PUBLIC_URL (optional)
"""

import os
import sys
import requests
import boto3
from botocore.config import Config as BotoConfig
from urllib.parse import urlparse
import time

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Database setup
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

DATABASE_URL = os.environ.get('DATABASE_URL', '')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Image model (simplified)
class Image(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    original_url = db.Column(db.String(500))
    processed_url = db.Column(db.String(500))
    original_filename = db.Column(db.String(255))

# R2 Configuration
R2_ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY')
R2_BUCKET_NAME = os.environ.get('R2_BUCKET_NAME', 'smartpneu-images')
R2_ACCOUNT_ID = os.environ.get('R2_ACCOUNT_ID', 'e276b85cfe3f10140f3eaa752e405d39')
R2_PUBLIC_URL = os.environ.get('R2_PUBLIC_URL', 'https://pub-7541f1c3a1ac4cd1a1af891329a72c80.r2.dev')

def get_r2_client():
    """Create R2/S3 client."""
    endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=BotoConfig(signature_version='s3v4'),
        region_name='auto'
    )

def is_cloudinary_url(url):
    """Check if URL is from Cloudinary."""
    if not url:
        return False
    return 'cloudinary.com' in url or 'res.cloudinary.com' in url

def download_image(url):
    """Download image from URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        return None

def get_extension_from_url(url):
    """Extract file extension from URL."""
    path = urlparse(url).path
    if '.' in path:
        return path.rsplit('.', 1)[-1].lower()
    return 'jpg'

def upload_to_r2(client, image_data, folder, original_url):
    """Upload image to R2 and return new URL."""
    import uuid
    from datetime import datetime
    
    ext = get_extension_from_url(original_url)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:8]
    object_key = f"{folder}/{timestamp}_{unique_id}.{ext}"
    
    content_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }
    content_type = content_types.get(ext, 'image/jpeg')
    
    client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=object_key,
        Body=image_data,
        ContentType=content_type
    )
    
    return f"{R2_PUBLIC_URL.rstrip('/')}/{object_key}"

def migrate_image_url(client, url, folder):
    """Migrate a single image URL from Cloudinary to R2."""
    if not is_cloudinary_url(url):
        return url  # Already R2 or other, skip
    
    # Download from Cloudinary
    image_data = download_image(url)
    if not image_data:
        return url  # Keep original on failure
    
    # Upload to R2
    try:
        new_url = upload_to_r2(client, image_data, folder, url)
        return new_url
    except Exception as e:
        print(f"  ERROR uploading to R2: {e}")
        return url  # Keep original on failure

def main():
    print("=" * 60)
    print("CLOUDINARY TO R2 MIGRATION")
    print("=" * 60)
    
    # Verify R2 credentials
    if not R2_ACCESS_KEY_ID or not R2_SECRET_ACCESS_KEY:
        print("ERROR: R2 credentials not set!")
        print("Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY environment variables")
        sys.exit(1)
    
    print(f"\nR2 Bucket: {R2_BUCKET_NAME}")
    print(f"R2 Public URL: {R2_PUBLIC_URL}")
    
    # Create R2 client
    r2_client = get_r2_client()
    
    # Test R2 connection
    try:
        r2_client.head_bucket(Bucket=R2_BUCKET_NAME)
        print("✓ R2 connection successful\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to R2: {e}")
        sys.exit(1)
    
    with app.app_context():
        # Get all images
        images = Image.query.all()
        total = len(images)
        
        # Count Cloudinary URLs
        cloudinary_original = sum(1 for img in images if is_cloudinary_url(img.original_url))
        cloudinary_processed = sum(1 for img in images if is_cloudinary_url(img.processed_url))
        
        print(f"Total images in database: {total}")
        print(f"Cloudinary original URLs: {cloudinary_original}")
        print(f"Cloudinary processed URLs: {cloudinary_processed}")
        print(f"Total to migrate: {cloudinary_original + cloudinary_processed}")
        print()
        
        if cloudinary_original + cloudinary_processed == 0:
            print("Nothing to migrate! All images already on R2.")
            return
        
        # Confirm
        confirm = input("Start migration? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            return
        
        print("\nStarting migration...\n")
        
        migrated_original = 0
        migrated_processed = 0
        failed = 0
        
        for i, img in enumerate(images, 1):
            changed = False
            
            # Migrate original_url
            if is_cloudinary_url(img.original_url):
                print(f"[{i}/{total}] Image {img.id} - original...")
                new_url = migrate_image_url(r2_client, img.original_url, 'originals')
                if new_url != img.original_url:
                    img.original_url = new_url
                    migrated_original += 1
                    changed = True
                    print(f"  ✓ Migrated to R2")
                else:
                    failed += 1
                    print(f"  ✗ Failed, keeping Cloudinary URL")
                time.sleep(0.1)  # Rate limiting
            
            # Migrate processed_url
            if is_cloudinary_url(img.processed_url):
                print(f"[{i}/{total}] Image {img.id} - processed...")
                new_url = migrate_image_url(r2_client, img.processed_url, 'processed')
                if new_url != img.processed_url:
                    img.processed_url = new_url
                    migrated_processed += 1
                    changed = True
                    print(f"  ✓ Migrated to R2")
                else:
                    failed += 1
                    print(f"  ✗ Failed, keeping Cloudinary URL")
                time.sleep(0.1)  # Rate limiting
            
            # Save after each image
            if changed:
                db.session.commit()
        
        print("\n" + "=" * 60)
        print("MIGRATION COMPLETE")
        print("=" * 60)
        print(f"Original images migrated: {migrated_original}")
        print(f"Processed images migrated: {migrated_processed}")
        print(f"Failed: {failed}")
        print()
        
        # Final check
        remaining = sum(1 for img in Image.query.all() 
                       if is_cloudinary_url(img.original_url) or is_cloudinary_url(img.processed_url))
        if remaining > 0:
            print(f"⚠ WARNING: {remaining} Cloudinary URLs still remain (failed migrations)")
        else:
            print("✓ All images successfully migrated to R2!")

if __name__ == '__main__':
    main()
