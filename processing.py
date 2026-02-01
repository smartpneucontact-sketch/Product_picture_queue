import requests
from PIL import Image, ImageEnhance
from io import BytesIO
import numpy as np

# Lazy load rembg (downloads model on first use)
_rembg_session = None

def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        _rembg_session = new_session("u2net")
    return _rembg_session


class ImageProcessor:
    def __init__(self, removebg_api_key=None, output_size=2048):
        # removebg_api_key kept for compatibility but not used
        self.output_size = output_size
        
        # Enhancement settings (tuned in the processing lab)
        self.brightness_factor = 1.10
        self.contrast_factor = 1.15
        self.sharpness_factor = 1.15
        self.shadow_lift = 20
        
        # Background removal settings
        self.use_alpha_matting = True
        self.erode_size = 12
        
        # Crop settings
        self.margin_percent = 4
    
    def enhance_image(self, img):
        """
        Apply enhancement to the tire image to show more detail.
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
        
        # 1. Shadow lifting (brightens dark areas to reveal tread detail)
        if self.shadow_lift > 0:
            img_array = np.array(enhanced, dtype=np.float32)
            shadow_mask = 1.0 - (img_array / 255.0)
            shadow_mask = shadow_mask ** 2  # More selective to truly dark areas
            lift_amount = self.shadow_lift * shadow_mask
            img_array = np.clip(img_array + lift_amount, 0, 255).astype(np.uint8)
            enhanced = Image.fromarray(img_array)
        
        # 2. Brightness adjustment
        if self.brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(self.brightness_factor)
        
        # 3. Contrast adjustment (brings out tread pattern)
        if self.contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(self.contrast_factor)
        
        # 4. Sharpness (crisp edges)
        if self.sharpness_factor != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(self.sharpness_factor)
        
        # Composite enhanced foreground onto pure white background
        if has_alpha and alpha_array is not None:
            white_bg = Image.new('RGB', enhanced.size, (255, 255, 255))
            alpha_mask = Image.fromarray(alpha_array)
            result = Image.composite(enhanced, white_bg, alpha_mask)
            return result
        
        return enhanced
    
    def remove_background(self, image_data):
        """
        Remove background using rembg with alpha matting for clean edges.
        Returns PNG with transparent background.
        """
        from rembg import remove
        
        input_image = Image.open(BytesIO(image_data))
        session = get_rembg_session()
        
        if self.use_alpha_matting:
            output_image = remove(
                input_image,
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=self.erode_size,
            )
        else:
            output_image = remove(input_image, session=session)
        
        # Convert to bytes
        output_buffer = BytesIO()
        output_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
        return output_buffer.getvalue()
    
    def crop_to_square(self, image_data):
        """
        Crop image to 1:1 aspect ratio with the subject (tire) centered.
        Preserves alpha channel for proper background handling.
        """
        img = Image.open(BytesIO(image_data))
        
        # Ensure RGBA mode for transparency handling
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Get bounding box of non-transparent pixels
        bbox = img.getbbox()
        
        if bbox is None:
            raise Exception("No content found in image after background removal")
        
        # Crop to content
        img_cropped = img.crop(bbox)
        width, height = img_cropped.size
        
        # Determine the size needed for square
        max_dim = max(width, height)
        
        # Add margin
        margin = int(max_dim * (self.margin_percent / 100))
        square_size = max_dim + (margin * 2)
        
        # Create transparent canvas (alpha=0 for background)
        square_img = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))
        
        # Calculate position to center the cropped image
        x_offset = (square_size - width) // 2
        y_offset = (square_size - height) // 2
        
        # Paste with alpha mask
        square_img.paste(img_cropped, (x_offset, y_offset), img_cropped)
        
        # Resize if needed
        if square_size > self.output_size:
            square_img = square_img.resize((self.output_size, self.output_size), Image.Resampling.LANCZOS)
        
        return square_img
    
    def process(self, image_data):
        """
        Full processing pipeline:
        1. Remove background (with alpha matting)
        2. Crop to 1:1 with tire centered
        3. Enhance (shadow lift, brightness, contrast, sharpness)
        4. Output as high-quality JPEG
        """
        # Step 1: Remove background
        no_bg_image = self.remove_background(image_data)
        
        # Step 2: Crop to square (returns RGBA PIL Image)
        square_img = self.crop_to_square(no_bg_image)
        
        # Step 3: Enhance (handles alpha, outputs RGB on white)
        enhanced_img = self.enhance_image(square_img)
        
        # Step 4: Save as high-quality JPEG
        output = BytesIO()
        enhanced_img.save(output, format='JPEG', quality=98, subsampling=0)
        output.seek(0)
        
        return output.getvalue()
    
    def process_with_settings(self, image_data, settings):
        """
        Process with custom settings (used by the lab).
        """
        # Apply custom settings
        if 'brightness' in settings:
            self.brightness_factor = settings['brightness']
        if 'contrast' in settings:
            self.contrast_factor = settings['contrast']
        if 'sharpness' in settings:
            self.sharpness_factor = settings['sharpness']
        if 'shadow_lift' in settings:
            self.shadow_lift = settings['shadow_lift']
        if 'erode_size' in settings:
            self.erode_size = settings['erode_size']
        if 'alpha_matting' in settings:
            self.use_alpha_matting = settings['alpha_matting']
        if 'margin_percent' in settings:
            self.margin_percent = settings['margin_percent']
        if 'output_size' in settings:
            self.output_size = settings['output_size']
        
        return self.process(image_data)
