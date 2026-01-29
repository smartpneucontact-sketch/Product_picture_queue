import requests
from PIL import Image
from io import BytesIO

# Lazy load rembg (downloads model on first use)
_rembg_remove = None

def get_rembg():
    global _rembg_remove
    if _rembg_remove is None:
        from rembg import remove
        _rembg_remove = remove
    return _rembg_remove

class ImageProcessor:
    def __init__(self, removebg_api_key=None, output_size=1000):
        # removebg_api_key kept for compatibility but not used
        self.output_size = output_size  # Final image will be output_size x output_size
    
    def remove_background(self, image_data):
        """
        Remove background using rembg (free, local processing).
        Returns PNG with transparent background.
        """
        # Open image
        input_image = Image.open(BytesIO(image_data))
        
        # Remove background (lazy load rembg)
        remove = get_rembg()
        output_image = remove(input_image)
        
        # Convert to bytes
        output_buffer = BytesIO()
        output_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
        return output_buffer.getvalue()
    
    def crop_to_square(self, image_data):
        """
        Crop image to 1:1 aspect ratio with the subject (tire) as large as possible.
        - Finds the bounding box of non-transparent pixels
        - Centers and pads to square
        - Resizes to output_size
        """
        img = Image.open(BytesIO(image_data))
        
        # Ensure RGBA mode for transparency handling
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Get bounding box of non-transparent pixels
        bbox = img.getbbox()
        
        if bbox is None:
            # No content found, return as-is
            raise Exception("No content found in image after background removal")
        
        # Crop to content
        img_cropped = img.crop(bbox)
        width, height = img_cropped.size
        
        # Determine the size needed for square (use the larger dimension)
        max_dim = max(width, height)
        
        # Add a small margin (5%)
        margin = int(max_dim * 0.05)
        square_size = max_dim + (margin * 2)
        
        # Create new square image with white background
        square_img = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 255))
        
        # Calculate position to center the cropped image
        x_offset = (square_size - width) // 2
        y_offset = (square_size - height) // 2
        
        # Paste cropped image onto square canvas
        square_img.paste(img_cropped, (x_offset, y_offset), img_cropped)
        
        # Resize to final output size
        final_img = square_img.resize((self.output_size, self.output_size), Image.Resampling.LANCZOS)
        
        # Convert to RGB (white background) for JPEG output
        rgb_img = Image.new('RGB', final_img.size, (255, 255, 255))
        rgb_img.paste(final_img, mask=final_img.split()[3])  # Use alpha channel as mask
        
        # Save to bytes
        output = BytesIO()
        rgb_img.save(output, format='JPEG', quality=95)
        output.seek(0)
        
        return output.getvalue()
    
    def process(self, image_data):
        """
        Full processing pipeline:
        1. Remove background (free, local)
        2. Crop to 1:1 with tire as large as possible
        """
        # Step 1: Remove background
        no_bg_image = self.remove_background(image_data)
        
        # Step 2: Crop to square
        final_image = self.crop_to_square(no_bg_image)
        
        return final_image
