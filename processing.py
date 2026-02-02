import requests
from PIL import Image, ImageEnhance
from io import BytesIO
import numpy as np

# Lazy load rembg (downloads model on first use)
_rembg_remove = None

def get_rembg():
    global _rembg_remove
    if _rembg_remove is None:
        from rembg import remove
        _rembg_remove = remove
    return _rembg_remove


class ImageProcessor:
    def __init__(self, removebg_api_key=None, output_size=2048):
        self.output_size = output_size
        
        # Enhancement settings
        self.brightness_factor = 1.10
        self.contrast_factor = 1.15
        self.sharpness_factor = 1.15
        self.shadow_lift = 20
        
        # Crop settings
        self.margin_percent = 4
    
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
    
    def remove_background(self, image_data):
        """
        Remove background using rembg.
        Returns PNG bytes with transparent background.
        """
        input_image = Image.open(BytesIO(image_data))
        
        remove = get_rembg()
        output_image = remove(input_image)
        
        output_buffer = BytesIO()
        output_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
        return output_buffer.getvalue()
    
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
    
    def process(self, image_data):
        """Full processing pipeline."""
        # Step 1: Remove background
        no_bg_image = self.remove_background(image_data)
        
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
        # Apply settings
        self.brightness_factor = settings.get('brightness', 1.10)
        self.contrast_factor = settings.get('contrast', 1.15)
        self.sharpness_factor = settings.get('sharpness', 1.15)
        self.shadow_lift = settings.get('shadow_lift', 20)
        self.margin_percent = settings.get('margin_percent', 4)
        self.output_size = settings.get('output_size', 2048)
        
        remove_bg = settings.get('remove_bg', True)
        
        # Step 1: Background removal (optional)
        if remove_bg:
            no_bg_data = self.remove_background(image_data)
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
