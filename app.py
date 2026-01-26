from flask import Flask, request, jsonify, render_template, redirect, url_for
from datetime import datetime
from config import Config
from models import db, Image
from storage import GCSStorage
from processing import ImageProcessor
from shopify_client import ShopifyClient
import threading

app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db.init_app(app)

# Initialize services (lazy loading)
_storage = None
_processor = None
_shopify = None

def get_storage():
    global _storage
    if _storage is None:
        _storage = GCSStorage(
            Config.GCS_BUCKET_NAME,
            Config.GCS_CREDENTIALS_JSON
        )
    return _storage

def get_processor():
    global _processor
    if _processor is None:
        _processor = ImageProcessor(
            Config.REMOVEBG_API_KEY,
            Config.OUTPUT_IMAGE_SIZE
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
        
        # Upload to GCS
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
    """Delete an image from the queue and GCS."""
    image = Image.query.get_or_404(image_id)
    
    try:
        storage = get_storage()
        
        # Delete from GCS
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
    thread = threading.Thread(target=process_images, args=(image_ids, app._get_current_object()))
    thread.start()
    
    return jsonify({
        'success': True,
        'message': f'Processing {len(images)} images for SKU {sku}'
    })


def process_images(image_ids, app):
    """Background task to process images and upload to Shopify."""
    with app.app_context():
        processor = get_processor()
        storage = get_storage()
        shopify = get_shopify()
        
        images = Image.query.filter(Image.id.in_(image_ids)).all()
        
        if not images:
            return
        
        sku = images[0].sku
        processed_urls = []
        
        for image in images:
            try:
                image.status = 'processing'
                db.session.commit()
                
                # Download original image
                import requests
                response = requests.get(image.original_url)
                original_data = response.content
                
                # Process image (bg removal + crop)
                processed_data = processor.process(original_data)
                
                # Upload processed image to GCS
                processed_url = storage.upload_processed_image(processed_data, image.original_filename)
                
                image.processed_url = processed_url
                image.status = 'processed'
                image.processed_at = datetime.utcnow()
                db.session.commit()
                
                processed_urls.append(processed_url)
                
            except Exception as e:
                image.status = 'failed'
                image.error_message = str(e)
                db.session.commit()
        
        # Upload all processed images to Shopify
        if processed_urls:
            try:
                result = shopify.add_images_to_product_by_sku(sku, processed_urls)
                
                # Mark all as completed
                for image in images:
                    if image.status == 'processed':
                        image.status = 'completed'
                db.session.commit()
                
            except Exception as e:
                for image in images:
                    if image.status == 'processed':
                        image.status = 'failed'
                        image.error_message = f"Shopify upload failed: {str(e)}"
                db.session.commit()


@app.route('/api/images/<int:image_id>/retry', methods=['POST'])
def retry_image(image_id):
    """Retry processing a failed image."""
    image = Image.query.get_or_404(image_id)
    
    if image.status != 'failed':
        return jsonify({'error': 'Image is not in failed status'}), 400
    
    if not image.sku:
        return jsonify({'error': 'Image has no SKU assigned'}), 400
    
    image.status = 'assigned'
    image.error_message = None
    db.session.commit()
    
    # Start processing
    thread = threading.Thread(target=process_images, args=([image_id], app._get_current_object()))
    thread.start()
    
    return jsonify({'success': True})


# =============================================================================
# Web UI
# =============================================================================

@app.route('/')
def index():
    """Main queue management UI."""
    return render_template('queue.html')


@app.route('/health')
def health():
    """Health check endpoint for Railway."""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
