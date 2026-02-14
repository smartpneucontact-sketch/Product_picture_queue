from flask import Flask, request, jsonify, render_template, redirect, url_for
from datetime import datetime
from config import Config
from models import db, Image
from storage import R2Storage
from processing_v2 import ImageProcessor
from shopify_client import ShopifyClient
import threading
import numpy as np
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db.init_app(app)

# Auto-migrate: add image_type column if it doesn't exist
with app.app_context():
    try:
        from sqlalchemy import inspect, text
        inspector = inspect(db.engine)
        columns = [c['name'] for c in inspector.get_columns('images')]
        if 'image_type' not in columns:
            db.session.execute(text("ALTER TABLE images ADD COLUMN image_type VARCHAR(20) DEFAULT 'front'"))
            db.session.commit()
            logger.info("Added image_type column to images table")
    except Exception as e:
        logger.warning(f"Migration check failed (may be normal on first run): {e}")

# Initialize services (lazy loading)
_storage = None
_processor = None
_shopify = None

def get_storage():
    global _storage
    if _storage is None:
        _storage = R2Storage(
            Config.R2_ACCESS_KEY_ID,
            Config.R2_SECRET_ACCESS_KEY,
            Config.R2_BUCKET_NAME,
            Config.R2_ACCOUNT_ID,
            Config.R2_PUBLIC_URL
        )
    return _storage

def get_processor():
    global _processor
    if _processor is None:
        _processor = ImageProcessor(
            poof_api_key=Config.POOF_API_KEY,
            output_size=Config.OUTPUT_IMAGE_SIZE
        )
    return _processor

def get_shopify():
    global _shopify
    if _shopify is None:
        _shopify = ShopifyClient(
            Config.SHOPIFY_STORE_URL,
            Config.SHOPIFY_ACCESS_TOKEN
        )
    return _shopify

# Create tables
with app.app_context():
    db.create_all()


# =============================================================================
# API Endpoints for Raspberry Pi
# =============================================================================

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """
    Receive an image from the Raspberry Pi.
    Expects: multipart/form-data with 'image' file and 'secret' field
    """
    # Simple auth check
    secret = request.form.get('secret')
    if secret != Config.UPLOAD_SECRET:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read file data
        file_data = file.read()
        
        # Upload to Cloudinary
        storage = get_storage()
        original_url = storage.upload_image(file_data, file.filename)
        
        # Create queue entry
        image = Image(
            original_filename=file.filename,
            original_url=original_url,
            status='pending'
        )
        db.session.add(image)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'id': image.id,
            'url': original_url
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# API Endpoints for Queue Management
# =============================================================================

@app.route('/api/images', methods=['GET'])
def get_images():
    """Get all images in the queue, ordered by creation date."""
    status = request.args.get('status', None)
    
    query = Image.query.order_by(Image.created_at.asc())
    
    if status:
        query = query.filter(Image.status == status)
    
    images = query.all()
    return jsonify([img.to_dict() for img in images])


@app.route('/api/images/<int:image_id>', methods=['DELETE'])
def delete_image(image_id):
    """Delete an image from the queue and Cloudinary."""
    image = Image.query.get_or_404(image_id)
    
    try:
        storage = get_storage()
        
        # Delete from Cloudinary
        if image.original_url:
            storage.delete_image(image.original_url)
        if image.processed_url:
            storage.delete_image(image.processed_url)
        
        # Delete from database
        db.session.delete(image)
        db.session.commit()
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/images/manual-upload', methods=['POST'])
def manual_upload():
    """Manually upload an image through the web UI."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        file_data = file.read()
        
        storage = get_storage()
        original_url = storage.upload_image(file_data, file.filename)
        
        image = Image(
            original_filename=file.filename,
            original_url=original_url,
            status='pending'
        )
        db.session.add(image)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'id': image.id,
            'image': image.to_dict()
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/assign', methods=['POST'])
def assign_sku():
    """
    Assign a SKU to selected images and start processing.
    Expects: {'image_ids': [1, 2, 3], 'sku': 'ABC123'}
    """
    data = request.get_json()
    image_ids = data.get('image_ids', [])
    sku = data.get('sku', '').strip()
    
    if not image_ids:
        return jsonify({'error': 'No images selected'}), 400
    
    if not sku:
        return jsonify({'error': 'SKU is required'}), 400
    
    # Update images with SKU
    images = Image.query.filter(Image.id.in_(image_ids)).all()
    
    for image in images:
        image.sku = sku
        image.status = 'assigned'
        image.assigned_at = datetime.utcnow()
    
    db.session.commit()
    
    # Start processing in background thread
    thread = threading.Thread(target=process_images, args=(image_ids, app, True))
    thread.start()
    
    return jsonify({
        'success': True,
        'message': f'Processing {len(images)} images for SKU {sku}'
    })


@app.route('/api/draft', methods=['POST'])
def draft_sku():
    """
    Assign a SKU to selected images WITHOUT processing or uploading.
    Just saves the SKU as a draft for later processing.
    Expects: {'image_ids': [1, 2, 3], 'sku': 'ABC123'}
    """
    data = request.get_json()
    image_ids = data.get('image_ids', [])
    sku = data.get('sku', '').strip()
    
    if not image_ids:
        return jsonify({'error': 'No images selected'}), 400
    
    if not sku:
        return jsonify({'error': 'SKU is required'}), 400
    
    # Update images with SKU but mark as draft
    images = Image.query.filter(Image.id.in_(image_ids)).all()
    
    for image in images:
        image.sku = sku
        image.status = 'draft'
        image.assigned_at = datetime.utcnow()
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': f'Saved {len(images)} images as draft with SKU {sku}'
    })


@app.route('/api/process-drafts', methods=['POST'])
def process_drafts():
    """
    Process draft images and upload to Shopify.
    Expects: {'image_ids': [1, 2, 3]} or {'sku': 'ABC123'}
    """
    data = request.get_json()
    image_ids = data.get('image_ids', [])
    sku = data.get('sku', '').strip()
    
    if image_ids:
        images = Image.query.filter(Image.id.in_(image_ids), Image.status == 'draft').all()
    elif sku:
        images = Image.query.filter(Image.sku == sku, Image.status == 'draft').all()
    else:
        return jsonify({'error': 'Provide image_ids or sku'}), 400
    
    if not images:
        return jsonify({'error': 'No draft images found'}), 400
    
    image_ids = [img.id for img in images]
    
    for image in images:
        image.status = 'assigned'
    db.session.commit()
    
    # Start processing in background thread
    thread = threading.Thread(target=process_images, args=(image_ids, app, True))
    thread.start()
    
    return jsonify({
        'success': True,
        'message': f'Processing {len(images)} draft images'
    })


@app.route('/api/gauge-crop', methods=['POST'])
def gauge_crop():
    """
    Detect tread depth gauge in an image and create a close-up crop.
    Expects: {'image_id': 1} or {'image_id': 1, 'x': 500, 'y': 300}
    
    If x,y provided: crop around that point (manual mode)
    If not: auto-detect the gauge position
    
    Creates a new image record with the cropped closeup.
    """
    data = request.get_json()
    image_id = data.get('image_id')
    manual_x = data.get('x')
    manual_y = data.get('y')
    
    if not image_id:
        return jsonify({'error': 'image_id required'}), 400
    
    image = Image.query.get(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404
    
    try:
        # Download the original image
        import requests as req
        response = req.get(image.original_url)
        original_data = response.content
        
        processor = get_processor()
        
        # Determine gauge position
        gauge_xy = None
        if manual_x is not None and manual_y is not None:
            gauge_xy = (int(manual_x), int(manual_y))
        
        # Create closeup
        result, message = processor.crop_gauge_closeup(original_data, gauge_xy=gauge_xy)
        
        if result is None:
            return jsonify({'error': message}), 400
        
        # Upload to Cloudinary
        storage = get_storage()
        closeup_filename = f"gauge_{image.original_filename}"
        closeup_url = storage.upload_processed_image(result, closeup_filename)
        
        # Update the existing image with the gauge closeup as processed version
        image.processed_url = closeup_url
        image.image_type = 'gauge'  # Mark as gauge so it won't be reprocessed
        image.status = 'processed'
        image.processed_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': message,
            'closeup_url': closeup_url,
            'image_id': image.id
        })
        
    except Exception as e:
        logger.error(f"Gauge crop failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/images/<int:image_id>/crop-region', methods=['POST'])
def set_crop_region(image_id):
    """
    Set a crop region for an image (to exclude people/unwanted objects).
    Expects: {'x': 0, 'y': 0, 'width': 100, 'height': 100} as percentages (0-100)
    """
    image = Image.query.get_or_404(image_id)
    data = request.get_json()
    
    x = data.get('x', 0)
    y = data.get('y', 0)
    width = data.get('width', 100)
    height = data.get('height', 100)
    
    # Store as JSON string in error_message field temporarily (or add new field)
    # Format: "crop:x,y,w,h"
    crop_data = f"crop:{x},{y},{width},{height}"
    
    # Clear any previous processed version so it gets reprocessed with crop
    image.processed_url = None
    image.status = 'pending'
    
    # Store crop region (using a simple format in the filename for now)
    # In production, add a crop_region column to the model
    if not image.original_filename.startswith('CROP_'):
        image.original_filename = f"CROP_{x}_{y}_{width}_{height}_" + image.original_filename
    else:
        # Update existing crop
        parts = image.original_filename.split('_', 5)
        image.original_filename = f"CROP_{x}_{y}_{width}_{height}_" + parts[5]
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'crop': {'x': x, 'y': y, 'width': width, 'height': height}
    })


@app.route('/api/images/<int:image_id>/clear-crop', methods=['POST'])
def clear_crop_region(image_id):
    """Clear crop region from an image."""
    image = Image.query.get_or_404(image_id)
    
    if image.original_filename.startswith('CROP_'):
        parts = image.original_filename.split('_', 5)
        if len(parts) >= 6:
            image.original_filename = parts[5]
        db.session.commit()
    
    return jsonify({'success': True})


@app.route('/api/gauge-detect', methods=['POST'])
def gauge_detect():
    """
    Detect gauge position in an image without cropping.
    Returns the detected position for UI overlay.
    Expects: {'image_id': 1}
    """
    data = request.get_json()
    image_id = data.get('image_id')
    
    if not image_id:
        return jsonify({'error': 'image_id required'}), 400
    
    image = Image.query.get(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404
    
    try:
        import requests as req
        response = req.get(image.original_url)
        original_data = response.content
        
        from PIL import Image as PILImage
        from io import BytesIO
        img = PILImage.open(BytesIO(original_data)).convert('RGB')
        img_array = np.array(img)
        
        processor = get_processor()
        result = processor.detect_gauge(img_array)
        
        if result is None:
            return jsonify({
                'detected': False,
                'message': 'No gauge found'
            })
        
        cx, cy, conf = result[0], result[1], result[2]
        gauge_type = result[3] if len(result) > 3 else 'digital'
        return jsonify({
            'detected': True,
            'x': cx,
            'y': cy,
            'confidence': round(conf, 2),
            'gauge_type': gauge_type,
            'image_width': img.size[0],
            'image_height': img.size[1]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process', methods=['POST'])
def process_only():
    """
    Process selected images without SKU assignment (no Shopify upload).
    Expects: {'image_ids': [1, 2, 3]}
    """
    data = request.get_json()
    image_ids = data.get('image_ids', [])
    
    if not image_ids:
        return jsonify({'error': 'No images selected'}), 400
    
    # Update images status
    images = Image.query.filter(Image.id.in_(image_ids)).all()
    
    for image in images:
        image.status = 'assigned'  # Reuse status for processing queue
    
    db.session.commit()
    
    # Start processing in background thread (without Shopify upload)
    thread = threading.Thread(target=process_images, args=(image_ids, app, False))
    thread.start()
    
    return jsonify({
        'success': True,
        'message': f'Processing {len(images)} images'
    })


def process_images(image_ids, app, upload_to_shopify=True):
    """Background task to process images and optionally upload to Shopify."""
    with app.app_context():
        processor = get_processor()
        storage = get_storage()
        
        # Fetch images and preserve the order from frontend
        images_map = {img.id: img for img in Image.query.filter(Image.id.in_(image_ids)).all()}
        images = [images_map[id] for id in image_ids if id in images_map]
        
        if not images:
            return
        
        sku = images[0].sku
        processed_urls = []
        
        for image in images:
            try:
                # Skip processing if already has a processed image
                if image.processed_url:
                    logger.info(f"Image {image.id} already processed, using existing")
                    processed_urls.append(image.processed_url)
                    if image.status != 'processed':
                        image.status = 'processed'
                        db.session.commit()
                    continue
                
                image.status = 'processing'
                db.session.commit()
                
                # Download original image
                import requests
                response = requests.get(image.original_url, timeout=30)
                if response.status_code != 200:
                    raise Exception(f"Failed to download image: HTTP {response.status_code}")
                if len(response.content) < 1000:
                    raise Exception(f"Downloaded data too small ({len(response.content)} bytes), likely an error page")
                original_data = response.content
                
                # Apply crop region if set (filename starts with CROP_x_y_w_h_)
                if image.original_filename.startswith('CROP_'):
                    try:
                        parts = image.original_filename.split('_')
                        crop_x = float(parts[1])  # percentages
                        crop_y = float(parts[2])
                        crop_w = float(parts[3])
                        crop_h = float(parts[4])
                        
                        from PIL import Image as PILImage
                        from io import BytesIO
                        
                        pil_img = PILImage.open(BytesIO(original_data))
                        img_w, img_h = pil_img.size
                        
                        # Convert percentages to pixels
                        left = int(img_w * crop_x / 100)
                        top = int(img_h * crop_y / 100)
                        right = int(img_w * (crop_x + crop_w) / 100)
                        bottom = int(img_h * (crop_y + crop_h) / 100)
                        
                        cropped = pil_img.crop((left, top, right, bottom))
                        
                        # Convert back to bytes
                        buffer = BytesIO()
                        cropped.save(buffer, format='JPEG', quality=98)
                        buffer.seek(0)
                        original_data = buffer.getvalue()
                        
                        logger.info(f"Applied crop region: {crop_x},{crop_y} {crop_w}x{crop_h}%")
                    except Exception as crop_err:
                        logger.warning(f"Failed to apply crop region: {crop_err}")
                
                # Process image (bg removal + crop)
                # Side images skip label detection, front images find max 1 label
                image_type = image.image_type or 'front'
                processed_data = processor.process(original_data, image_type=image_type)
                
                # Upload processed image to Cloudinary
                processed_url = storage.upload_processed_image(processed_data, image.original_filename)
                
                image.processed_url = processed_url
                image.status = 'processed'
                image.processed_at = datetime.utcnow()
                db.session.commit()
                
                processed_urls.append(processed_url)
                
            except Exception as e:
                logger.error(f"Processing failed for image {image.id}: {e}")
                image.status = 'failed'
                image.error_message = str(e)
                db.session.commit()
        
        # Upload all processed images to Shopify (only if SKU assigned and flag is True)
        if upload_to_shopify and sku and processed_urls:
            try:
                shopify = get_shopify()
                result = shopify.add_images_to_product_by_sku(sku, processed_urls)
                
                # Mark all as completed
                for image in images:
                    if image.status == 'processed':
                        image.status = 'completed'
                db.session.commit()
                
            except Exception as e:
                logger.error(f"Shopify upload failed for SKU {sku}: {e}")
                for image in images:
                    if image.status == 'processed':
                        image.status = 'failed'
                        image.error_message = f"Shopify upload failed: {str(e)}"
                db.session.commit()
        # If not uploading to Shopify, leave status as 'processed' (not 'completed')


@app.route('/api/images/<int:image_id>/retry', methods=['POST'])
def retry_image(image_id):
    """Retry processing a failed image."""
    image = Image.query.get_or_404(image_id)
    
    if image.status != 'failed':
        return jsonify({'error': 'Image is not in failed status'}), 400
    
    image.status = 'assigned'
    image.error_message = None
    db.session.commit()
    
    # Start processing (upload to Shopify only if SKU is assigned)
    upload_to_shopify = bool(image.sku)
    thread = threading.Thread(target=process_images, args=([image_id], app, upload_to_shopify))
    thread.start()
    
    return jsonify({'success': True})


@app.route('/api/images/<int:image_id>/reprocess', methods=['POST'])
def reprocess_queue_image(image_id):
    """Clear processed version and reprocess from original."""
    image = Image.query.get_or_404(image_id)
    
    # Clear processed version
    image.processed_url = None
    image.status = 'assigned'
    image.error_message = None
    db.session.commit()
    
    # Start processing
    thread = threading.Thread(target=process_images, args=([image_id], app, False))
    thread.start()
    
    return jsonify({'success': True})


@app.route('/api/reset-stuck', methods=['GET', 'POST'])
def reset_stuck():
    """Reset stuck processing/assigned images to failed status."""
    stuck = Image.query.filter(Image.status.in_(['processing', 'assigned'])).all()
    count = 0
    for img in stuck:
        img.status = 'failed'
        img.error_message = 'Reset: was stuck in processing'
        count += 1
    db.session.commit()
    logger.info(f"Reset {count} stuck images")
    return jsonify({'reset': count})


@app.route('/api/images/<int:image_id>/mark-side', methods=['POST'])
def mark_side(image_id):
    """Toggle image type between front and side."""
    image = Image.query.get_or_404(image_id)
    
    # Toggle between front and side
    if image.image_type == 'side':
        image.image_type = 'front'
    else:
        image.image_type = 'side'
    
    db.session.commit()
    return jsonify({'success': True, 'image_type': image.image_type})


# =============================================================================
# Web UI
# =============================================================================

@app.route('/')
def index():
    """Main queue management UI."""
    return render_template('queue.html')


@app.route('/lab')
def lab():
    """Processing Lab UI - reprocess images with different settings."""
    return render_template('lab.html')


@app.route('/health')
def health():
    """Health check endpoint for Railway."""
    return jsonify({'status': 'ok'})


# =============================================================================
# Processing Lab API
# =============================================================================

@app.route('/api/lab/days')
def get_lab_days():
    """Get list of days with image counts for the lab."""
    from sqlalchemy import func
    
    results = db.session.query(
        func.date(Image.created_at).label('date'),
        func.count(Image.id).label('count')
    ).group_by(
        func.date(Image.created_at)
    ).order_by(
        func.date(Image.created_at).desc()
    ).all()
    
    total_images = db.session.query(func.count(Image.id)).scalar() or 0
    
    days = [{'date': str(r.date), 'count': r.count} for r in results]
    
    return jsonify({
        'days': days,
        'total_images': total_images
    })


@app.route('/api/lab/images/day/<date>')
def get_lab_images_by_day(date):
    """Get all images for a specific day (lazy loading for the lab)."""
    from sqlalchemy import func
    
    images = Image.query.filter(
        func.date(Image.created_at) == date
    ).order_by(
        Image.created_at.asc()
    ).all()
    
    return jsonify([img.to_dict() for img in images])


@app.route('/api/lab/reprocess/<int:image_id>', methods=['POST'])
def reprocess_image(image_id):
    """Reprocess a single image with custom settings from the lab."""
    import time
    import requests as http_requests
    
    start_time = time.time()
    
    image = Image.query.get_or_404(image_id)
    settings = request.get_json() or {}
    
    try:
        # Download original image
        response = http_requests.get(image.original_url)
        response.raise_for_status()
        original_data = response.content
        
        # Process with custom settings
        processor = ImageProcessor(
            poof_api_key=Config.POOF_API_KEY,
            output_size=settings.get('output_size', Config.OUTPUT_IMAGE_SIZE)
        )
        processed_data = processor.process_with_settings(original_data, settings)
        
        # Upload to Cloudinary if enabled
        if settings.get('save_to_cloud', True):
            storage = get_storage()
            processed_url = storage.upload_processed_image(processed_data, image.original_filename)
            
            # Update database
            image.processed_url = processed_url
            image.processed_at = datetime.utcnow()
            if image.status == 'pending':
                image.status = 'processed'
            db.session.commit()
        else:
            # Return base64 for preview only
            import base64
            processed_url = 'data:image/jpeg;base64,' + base64.b64encode(processed_data).decode('utf-8')
        
        elapsed = time.time() - start_time
        
        return jsonify({
            'success': True,
            'processed_url': processed_url,
            'time': elapsed
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
