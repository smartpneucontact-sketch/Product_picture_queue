# SmartPneu Studio

Image processing pipeline for tire product photography.

## Flow

```
Camera → Raspberry Pi → This App → remove.bg → GCS → Shopify
```

## Features

- 📷 Receive images from Raspberry Pi studio setup
- 📋 Queue management with chronological ordering
- ✂️ Automatic background removal (remove.bg)
- 🔲 Crop to 1:1 aspect ratio with tire maximized
- 🛒 Auto-upload to Shopify products by SKU
- 🖼️ Manual image upload support
- 🗑️ Delete images from queue

## Setup

### 1. Google Cloud Storage

1. Create a GCS bucket (e.g., `smartpneu-studio`)
2. Create a service account with Storage Admin role
3. Download the JSON key file
4. Set bucket permissions to allow public read (for Shopify to access images)

### 2. remove.bg

1. Sign up at https://www.remove.bg/
2. Get your API key from the dashboard

### 3. Shopify

1. Create a private app in Shopify admin
2. Enable read/write access for Products
3. Copy the access token

### 4. Deploy to Railway

1. Connect your GitHub repo to Railway
2. Add a PostgreSQL database
3. Set environment variables (see `.env.example`)

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (auto-set by Railway) |
| `GCS_BUCKET_NAME` | Your GCS bucket name |
| `GCS_CREDENTIALS_JSON` | Service account JSON (as single-line string) |
| `REMOVEBG_API_KEY` | Your remove.bg API key |
| `SHOPIFY_STORE_URL` | Your store URL (e.g., `smartpneu.myshopify.com`) |
| `SHOPIFY_ACCESS_TOKEN` | Shopify private app access token |
| `UPLOAD_SECRET` | Secret key for Pi authentication |

## API Endpoints

### Upload (for Raspberry Pi)

```bash
POST /api/upload
Content-Type: multipart/form-data

image: <file>
secret: <UPLOAD_SECRET>
```

### Queue Management

```bash
GET /api/images                    # List all images
GET /api/images?status=pending     # Filter by status
DELETE /api/images/<id>            # Delete image
POST /api/images/manual-upload     # Upload via web UI
POST /api/assign                   # Assign SKU and process
POST /api/images/<id>/retry        # Retry failed image
```

## Raspberry Pi Script

See the `pi/` folder for the camera capture and upload script.

```python
# Minimal example
import requests
import gphoto2 as gp

UPLOAD_URL = "https://your-app.railway.app/api/upload"
SECRET = "your-secret"

camera = gp.Camera()
camera.init()

# Capture and upload
file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
camera_file = camera.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
file_data = camera_file.get_data_and_size()

requests.post(UPLOAD_URL, files={'image': file_data}, data={'secret': SECRET})
```

## Local Development

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
python app.py
```

## Processing Pipeline

1. **Receive** - Image uploaded from Pi or manually
2. **Queue** - Stored in GCS, added to pending queue
3. **Assign** - User assigns SKU via web UI
4. **Process** - Background removal + square crop
5. **Upload** - Processed images sent to Shopify product
6. **Complete** - Status updated, visible in completed tab
