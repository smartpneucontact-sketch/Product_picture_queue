import requests
from PIL import Image, ImageEnhance
from io import BytesIO
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# Lazy load rembg (downloads model on first use)
_rembg_remove = None

def get_rembg():
    global _rembg_remove
    if _rembg_remove is None:
        from rembg import remove
        _rembg_remove = remove
    return _rembg_remove


class ImageProcessor:
    def __init__(self, poof_api_key=None, output_size=2048):
        self.output_size = output_size
        self.poof_api_key = poof_api_key or os.environ.get('POOF_API_KEY')
        
        # Enhancement settings (matching Lab2)
        self.brightness_factor = 1.15
        self.contrast_factor = 1.20
        self.sharpness_factor = 1.10
        self.shadow_lift = 20
        
        # Crop settings
        self.margin_percent = 5
        
        # Alpha matting settings for rembg (disabled by default - too slow)
        # Enable only in Lab for testing
        self.alpha_matting = False
        self.erode_size = 10
        self.fg_threshold = 240
        self.bg_threshold = 10
        
        # Background removal method: 'poof' (fast), 'rembg', or 'auto'
        # Use Poof API by default if key is available (fast + high quality)
        self.bg_removal_method = 'poof' if self.poof_api_key else 'rembg'
    
    def remove_background_poof(self, image_data):
        """
        Remove background using Poof API (removebgapi.com).
        Returns PNG bytes with transparent background.
        """
        if not self.poof_api_key:
            raise ValueError("Poof API key not configured")
        
        logger.info("Removing background using Poof API...")
        
        # Poof API endpoint
        url = "https://api.poof.bg/v1/remove-background"
        
        headers = {
            "X-Api-Key": self.poof_api_key
        }
        
        files = {
            'image_file': ('image.jpg', image_data, 'image/jpeg')
        }
        
        try:
            response = requests.post(url, headers=headers, files=files, timeout=60)
            
            if response.status_code == 200:
                logger.info(f"Poof API success - received {len(response.content)} bytes")
                return response.content
            else:
                logger.error(f"Poof API error: {response.status_code} - {response.text[:200]}")
                raise Exception(f"Poof API error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.error("Poof API timeout")
            raise Exception("Poof API timeout - try again later")
        except requests.exceptions.RequestException as e:
            logger.error(f"Poof API request error: {e}")
            raise Exception(f"Poof API error: {e}")
    
    def remove_background_rembg(self, image_data):
        """
        Remove background using rembg (local, free) with alpha matting.
        Returns PNG bytes with transparent background.
        """
        logger.info("Removing background using rembg with alpha_matting...")
        input_image = Image.open(BytesIO(image_data))
        
        remove = get_rembg()
        
        # Use alpha matting for better edge quality (like Lab2)
        if self.alpha_matting:
            from rembg import new_session
            session = new_session("u2net")
            output_image = remove(
                input_image,
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=self.fg_threshold,
                alpha_matting_background_threshold=self.bg_threshold,
                alpha_matting_erode_size=self.erode_size,
            )
        else:
            output_image = remove(input_image)
        
        output_buffer = BytesIO()
        output_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
        return output_buffer.getvalue()
    
    def enhance_image(self, img):
        """
        Apply enhancement to the tire image.
        Only enhances the foreground, preserving white background.
        """
        has_alpha = img.mode == 'RGBA'
        
        if has_alpha:
            r, g, b, a = img.split()
            rgb_img = Image.merge('RGB', (r, g, b))
            alpha_array = np.array(a)
        else:
            rgb_img = img.convert('RGB')
            alpha_array = None
        
        enhanced = rgb_img.copy()
        
        # 1. Shadow lifting
        if self.shadow_lift > 0:
            img_array = np.array(enhanced, dtype=np.float32)
            shadow_mask = 1.0 - (img_array / 255.0)
            shadow_mask = shadow_mask ** 2
            lift_amount = self.shadow_lift * shadow_mask
            img_array = np.clip(img_array + lift_amount, 0, 255).astype(np.uint8)
            enhanced = Image.fromarray(img_array)
        
        # 2. Brightness
        if self.brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(self.brightness_factor)
        
        # 3. Contrast
        if self.contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(self.contrast_factor)
        
        # 4. Sharpness
        if self.sharpness_factor != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(self.sharpness_factor)
        
        # Composite onto white background
        if has_alpha and alpha_array is not None:
            white_bg = Image.new('RGB', enhanced.size, (255, 255, 255))
            alpha_mask = Image.fromarray(alpha_array)
            result = Image.composite(enhanced, white_bg, alpha_mask)
            return result
        
        return enhanced
    
    def remove_background(self, image_data, method=None):
        """
        Remove background using configured method.
        Methods: 'poof', 'rembg', or 'auto' (try poof first, fallback to rembg)
        Returns PNG bytes with transparent background.
        """
        method = method or self.bg_removal_method
        
        if method == 'poof':
            try:
                return self.remove_background_poof(image_data)
            except Exception as e:
                logger.warning(f"Poof failed, falling back to rembg: {e}")
                return self.remove_background_rembg(image_data)
        elif method == 'rembg':
            return self.remove_background_rembg(image_data)
        elif method == 'auto':
            # Try Poof first if API key is available
            if self.poof_api_key:
                try:
                    return self.remove_background_poof(image_data)
                except Exception as e:
                    logger.warning(f"Poof failed, falling back to rembg: {e}")
            return self.remove_background_rembg(image_data)
        else:
            raise ValueError(f"Unknown background removal method: {method}")
    
    def crop_to_square(self, image_data):
        """
        Crop image to 1:1 aspect ratio with the subject centered.
        """
        if isinstance(image_data, bytes):
            img = Image.open(BytesIO(image_data))
        else:
            img = image_data
        
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Get bounding box of non-transparent pixels
        bbox = img.getbbox()
        
        if bbox is None:
            raise Exception("No content found in image")
        
        # Crop to content
        img_cropped = img.crop(bbox)
        width, height = img_cropped.size
        
        # Square size with margin
        max_dim = max(width, height)
        margin = int(max_dim * (self.margin_percent / 100))
        square_size = max_dim + (margin * 2)
        
        # Create transparent canvas
        square_img = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))
        
        # Center the cropped image
        x_offset = (square_size - width) // 2
        y_offset = (square_size - height) // 2
        square_img.paste(img_cropped, (x_offset, y_offset), img_cropped)
        
        # Resize if needed
        if square_size > self.output_size:
            square_img = square_img.resize((self.output_size, self.output_size), Image.Resampling.LANCZOS)
        
        return square_img
    
    def process(self, image_data, bg_method=None):
        """Full processing pipeline."""
        # Step 1: Remove background
        no_bg_image = self.remove_background(image_data, method=bg_method)
        
        # Step 2: Crop to square
        square_img = self.crop_to_square(no_bg_image)
        
        # Step 3: Enhance
        enhanced_img = self.enhance_image(square_img)
        
        # Step 4: Save as JPEG
        output = BytesIO()
        enhanced_img.save(output, format='JPEG', quality=98, subsampling=0)
        output.seek(0)
        
        return output.getvalue()
    
    def process_with_settings(self, image_data, settings):
        """Process with custom settings (used by the lab)."""
        # Apply enhancement settings
        self.brightness_factor = settings.get('brightness', 1.15)
        self.contrast_factor = settings.get('contrast', 1.20)
        self.sharpness_factor = settings.get('sharpness', 1.10)
        self.shadow_lift = settings.get('shadow_lift', 20)
        self.margin_percent = settings.get('margin_percent', 5)
        self.output_size = settings.get('output_size', 2048)
        
        # Alpha matting settings
        self.alpha_matting = settings.get('alpha_matting', True)
        self.erode_size = settings.get('erode_size', 10)
        self.fg_threshold = settings.get('fg_threshold', 240)
        self.bg_threshold = settings.get('bg_threshold', 10)
        
        remove_bg = settings.get('remove_bg', True)
        bg_method = settings.get('bg_method', 'auto')  # 'poof', 'rembg', or 'auto'
        
        # Step 1: Background removal (optional)
        if remove_bg:
            no_bg_data = self.remove_background(image_data, method=bg_method)
            img = Image.open(BytesIO(no_bg_data))
        else:
            img = Image.open(BytesIO(image_data))
        
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Step 2: Crop to square
        square_img = self.crop_to_square(img)
        
        # Step 3: Enhance
        enhanced_img = self.enhance_image(square_img)
        
        # Step 4: Save as JPEG
        output = BytesIO()
        enhanced_img.save(output, format='JPEG', quality=98, subsampling=0)
        output.seek(0)
        
        return output.getvalue()
