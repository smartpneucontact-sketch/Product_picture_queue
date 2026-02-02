"""
SmartPneu Image Processing v2
Advanced tire image processing with intelligent label protection and tread enhancement.

Key improvements over v1:
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast
- Intelligent label detection using edge density analysis
- Separate processing pipelines for tire rubber vs label areas
- Better shadow recovery with tone mapping
- Optional color correction for accurate tire appearance
"""

import requests
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import numpy as np
import os
import logging
import cv2

logger = logging.getLogger(__name__)

# Lazy load rembg
_rembg_remove = None

def get_rembg():
    global _rembg_remove
    if _rembg_remove is None:
        from rembg import remove
        _rembg_remove = remove
    return _rembg_remove


class ImageProcessorV2:
    """Advanced tire image processor with intelligent region-based enhancement."""
    
    def __init__(self, poof_api_key=None, output_size=2048):
        self.output_size = output_size
        self.poof_api_key = poof_api_key or os.environ.get('POOF_API_KEY')
        
        # ===== ENHANCEMENT PRESETS =====
        # Tire rubber enhancement (applied to dark tire areas)
        self.tire_clahe_clip = 2.0       # CLAHE clip limit (higher = more contrast)
        self.tire_clahe_grid = 8         # CLAHE grid size
        self.tire_brightness = 1.05      # Subtle brightening for tire
        self.tire_contrast = 1.10        # Local contrast boost
        self.shadow_lift = 15            # Lift deep shadows (0-50)
        
        # Label protection (bright text areas)
        self.label_threshold = 200       # Brightness threshold for label detection
        self.label_edge_threshold = 50   # Edge density threshold for text areas
        self.label_protection = 1.0      # Full protection - preserve labels exactly as original
        
        # Global adjustments (applied to whole image)
        self.sharpness_factor = 1.15     # Overall sharpening
        self.color_correction = True     # Auto white balance
        self.denoise_strength = 3        # Noise reduction (0 = off)
        
        # Crop settings
        self.margin_percent = 5
        
        # Background removal
        self.bg_removal_method = 'poof' if self.poof_api_key else 'rembg'
        self.alpha_matting = False
        self.erode_size = 10
        self.fg_threshold = 240
        self.bg_threshold = 10
        
        # Edge refinement settings (NEW)
        self.edge_refinement = True      # Enable edge smoothing
        self.edge_feather = 2            # Feather radius in pixels (1-5)
        self.edge_smooth = 3             # Gaussian blur on mask edges
        self.edge_erode = 1              # Erode mask to remove fringe (0-3)
        self.anti_alias = True           # Anti-alias the final edges
    
    # ==================== BACKGROUND REMOVAL ====================
    
    def remove_background_poof(self, image_data):
        """Remove background using Poof API."""
        if not self.poof_api_key:
            raise ValueError("Poof API key not configured")
        
        logger.info("Removing background using Poof API...")
        url = "https://api.poof.bg/v1/remove-background"
        headers = {"X-Api-Key": self.poof_api_key}
        files = {'image_file': ('image.jpg', image_data, 'image/jpeg')}
        
        try:
            response = requests.post(url, headers=headers, files=files, timeout=60)
            if response.status_code == 200:
                logger.info(f"Poof API success - {len(response.content)} bytes")
                return response.content
            else:
                logger.error(f"Poof API error: {response.status_code}")
                raise Exception(f"Poof API error: {response.status_code}")
        except requests.exceptions.Timeout:
            raise Exception("Poof API timeout")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Poof API error: {e}")
    
    def remove_background_rembg(self, image_data):
        """Remove background using rembg."""
        logger.info("Removing background using rembg...")
        input_image = Image.open(BytesIO(image_data))
        remove = get_rembg()
        
        if self.alpha_matting:
            from rembg import new_session
            session = new_session("u2net")
            output_image = remove(
                input_image, session=session,
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
    
    def remove_background(self, image_data, method=None):
        """Remove background with fallback."""
        method = method or self.bg_removal_method
        
        if method in ['poof', 'auto'] and self.poof_api_key:
            try:
                return self.remove_background_poof(image_data)
            except Exception as e:
                logger.warning(f"Poof failed, falling back to rembg: {e}")
                return self.remove_background_rembg(image_data)
        return self.remove_background_rembg(image_data)
    
    # ==================== EDGE REFINEMENT ====================
    
    def refine_edges(self, img):
        """
        Refine the edges of a transparent PNG to remove jagged edges and fringing.
        
        Techniques:
        1. Erode the alpha mask slightly to remove color fringe
        2. Smooth the alpha mask edges with Gaussian blur
        3. Apply feathering for gradual transparency at edges
        4. Anti-alias by super-sampling edges
        """
        if img.mode != 'RGBA':
            return img
        
        if not self.edge_refinement:
            return img
        
        # Split into RGB and Alpha
        r, g, b, a = img.split()
        alpha = np.array(a, dtype=np.uint8)
        rgb = np.array(img.convert('RGB'))
        
        original_alpha = alpha.copy()
        
        # Step 1: Erode to remove color fringe (pixels at the very edge often have wrong colors)
        if self.edge_erode > 0:
            kernel = np.ones((3, 3), np.uint8)
            alpha = cv2.erode(alpha, kernel, iterations=self.edge_erode)
        
        # Step 2: Find edge region (where alpha transitions from 0 to 255)
        # Dilate and erode to find the edge band
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(alpha, kernel, iterations=2)
        eroded = cv2.erode(alpha, kernel, iterations=2)
        edge_band = dilated - eroded  # This is the edge region
        
        # Step 3: Smooth only the edge region
        if self.edge_smooth > 0:
            # Apply Gaussian blur to the alpha
            blur_size = self.edge_smooth * 2 + 1  # Must be odd
            alpha_smooth = cv2.GaussianBlur(alpha.astype(np.float32), (blur_size, blur_size), 0)
            
            # Blend: use smoothed alpha in edge regions, original elsewhere
            edge_mask = (edge_band > 0).astype(np.float32)
            edge_mask = cv2.GaussianBlur(edge_mask, (blur_size, blur_size), 0)
            
            alpha = (alpha_smooth * edge_mask + alpha.astype(np.float32) * (1 - edge_mask))
            alpha = np.clip(alpha, 0, 255).astype(np.uint8)
        
        # Step 4: Feathering - gradual falloff at edges
        if self.edge_feather > 0:
            # Distance transform from the edge
            binary_mask = (alpha > 127).astype(np.uint8) * 255
            dist_inside = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
            dist_outside = cv2.distanceTransform(255 - binary_mask, cv2.DIST_L2, 5)
            
            # Create feathered alpha based on distance
            feather_px = self.edge_feather
            feather_alpha = np.clip(dist_inside / feather_px, 0, 1)
            
            # Apply feathering only where original alpha was transitioning
            edge_region = (alpha > 10) & (alpha < 245)
            alpha_float = alpha.astype(np.float32) / 255.0
            alpha_float = np.where(edge_region, feather_alpha, alpha_float)
            alpha = (alpha_float * 255).astype(np.uint8)
        
        # Step 5: Anti-aliasing - ensure smooth gradients at edges
        if self.anti_alias:
            # Apply slight bilateral filter to smooth while preserving edges
            alpha = cv2.bilateralFilter(alpha, 5, 50, 50)
        
        # Step 6: Clean up any semi-transparent noise in fully opaque areas
        # Pixels that were fully opaque should stay fully opaque (except at edges)
        fully_opaque = original_alpha == 255
        fully_transparent = original_alpha == 0
        
        # Restore full opacity to core areas (not near edges)
        core_opaque = cv2.erode(fully_opaque.astype(np.uint8) * 255, kernel, iterations=3)
        alpha = np.where(core_opaque > 0, 255, alpha)
        
        # Restore full transparency to areas that were background
        core_transparent = cv2.erode(fully_transparent.astype(np.uint8) * 255, kernel, iterations=3)
        alpha = np.where(core_transparent > 0, 0, alpha)
        
        # Reconstruct the image
        result = Image.merge('RGBA', (r, g, b, Image.fromarray(alpha)))
        
        logger.debug(f"Edge refinement complete: erode={self.edge_erode}, smooth={self.edge_smooth}, feather={self.edge_feather}")
        
        return result
    
    # ==================== LABEL DETECTION ====================
    
    def detect_label_regions(self, img_array, alpha_mask=None):
        """
        Detect label/text regions on tire.
        Labels are typically white/light rectangles with dark text on the tire.
        
        Args:
            img_array: RGB image array
            alpha_mask: Optional alpha channel to identify tire vs background
            
        Returns a mask where 1.0 = label area (protect), 0.0 = tire area (enhance).
        """
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        h, w = gray.shape
        label_mask = np.zeros((h, w), dtype=np.float32)
        
        # If we have alpha mask, only look for labels ON the tire (not background)
        if alpha_mask is not None:
            tire_region = alpha_mask > 128
        else:
            # Assume non-white areas are tire
            tire_region = gray < 240
        
        # 1. Find bright regions within the tire (potential labels)
        # Labels are typically much brighter than tire rubber (dark gray/black)
        brightness = gray.astype(np.float32)
        
        # Calculate tire average brightness (excluding very bright areas)
        tire_pixels = brightness[tire_region & (brightness < 200)]
        if len(tire_pixels) > 0:
            tire_avg = np.mean(tire_pixels)
            tire_std = np.std(tire_pixels)
        else:
            tire_avg = 50
            tire_std = 30
        
        # Labels are significantly brighter than tire average
        label_brightness_threshold = min(tire_avg + 3 * tire_std, self.label_threshold)
        
        # Create mask of bright-on-tire regions
        bright_on_tire = (brightness > label_brightness_threshold) & tire_region
        bright_mask = bright_on_tire.astype(np.uint8) * 255
        
        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 2. Find contours of bright regions
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size - labels are typically 1-15% of tire area
            tire_area = np.sum(tire_region)
            min_area = tire_area * 0.002  # Min 0.2% of tire
            max_area = tire_area * 0.2    # Max 20% of tire
            
            if area < min_area or area > max_area:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Check aspect ratio - labels are usually rectangular
            aspect = max(cw, ch) / (min(cw, ch) + 1)
            if aspect > 8:  # Too elongated
                continue
            
            # Check if region has text-like contrast (dark on light)
            roi = gray[y:y+ch, x:x+cw]
            roi_std = np.std(roi)
            
            # Labels have high local contrast (text)
            if roi_std > 20:  # Has dark text on light background
                # Mark this region as label
                pad = 15
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(w, x + cw + pad), min(h, y + ch + pad)
                label_mask[y1:y2, x1:x2] = 1.0
                logger.debug(f"Label detected: ({x},{y}) {cw}x{ch}, std={roi_std:.1f}")
        
        # 3. Also protect pure white areas on the tire (may be label edges)
        pure_white_on_tire = (brightness > 230) & tire_region
        label_mask = np.maximum(label_mask, pure_white_on_tire.astype(np.float32))
        
        # 4. Smooth the mask edges for blending
        if np.any(label_mask > 0):
            label_mask = cv2.GaussianBlur(label_mask, (15, 15), 0)
            label_mask = np.clip(label_mask, 0, 1)
        
        coverage = np.mean(label_mask) * 100
        logger.info(f"Label detection: {coverage:.1f}% of image protected")
        
        return label_mask
    
    def detect_label_bbox(self, img_array):
        """
        Detect the main label bounding box for more precise protection.
        Returns (x, y, w, h) or None if no label found.
        """
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Threshold to find white regions
        _, thresh = cv2.threshold(gray, self.label_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_label = None
        best_score = 0
        
        h, w = gray.shape
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2000 or area > (h * w * 0.25):
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Score based on rectangularity and size
            rect_area = cw * ch
            rectangularity = area / (rect_area + 1)
            
            # Check for text content
            roi = gray[y:y+ch, x:x+cw]
            std_dev = np.std(roi)  # High std = has dark text on white
            
            score = rectangularity * std_dev * (area / 10000)
            
            if score > best_score:
                best_score = score
                best_label = (x, y, cw, ch)
        
        return best_label
    
    # ==================== ENHANCEMENT ====================
    
    def apply_clahe(self, img_array, clip_limit=2.0, grid_size=8):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        This enhances local contrast without blowing out bright areas.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        l_enhanced = clahe.apply(l)
        
        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        return rgb_enhanced
    
    def lift_shadows(self, img_array, amount=20):
        """
        Lift shadows while preserving highlights.
        Uses a tone curve that affects only dark areas.
        """
        img_float = img_array.astype(np.float32) / 255.0
        
        # Shadow mask: 1.0 for black, 0.0 for white
        luminance = 0.299 * img_float[:,:,0] + 0.587 * img_float[:,:,1] + 0.114 * img_float[:,:,2]
        shadow_mask = (1.0 - luminance) ** 2  # Squared for more aggressive dark targeting
        
        # Apply lift
        lift = (amount / 255.0) * shadow_mask
        lift = np.dstack([lift, lift, lift])
        
        result = np.clip(img_float + lift, 0, 1)
        return (result * 255).astype(np.uint8)
    
    def auto_white_balance(self, img_array):
        """
        Simple gray-world white balance correction.
        Ensures tire colors look natural, not too yellow/blue.
        """
        result = img_array.astype(np.float32)
        
        # Calculate average for each channel
        avg_r = np.mean(result[:,:,0])
        avg_g = np.mean(result[:,:,1])
        avg_b = np.mean(result[:,:,2])
        avg_gray = (avg_r + avg_g + avg_b) / 3
        
        # Scale each channel
        if avg_r > 0:
            result[:,:,0] = np.clip(result[:,:,0] * (avg_gray / avg_r), 0, 255)
        if avg_g > 0:
            result[:,:,1] = np.clip(result[:,:,1] * (avg_gray / avg_g), 0, 255)
        if avg_b > 0:
            result[:,:,2] = np.clip(result[:,:,2] * (avg_gray / avg_b), 0, 255)
        
        return result.astype(np.uint8)
    
    def denoise(self, img_array, strength=3):
        """Apply non-local means denoising."""
        if strength <= 0:
            return img_array
        return cv2.fastNlMeansDenoisingColored(img_array, None, strength, strength, 7, 21)
    
    def enhance_image(self, img):
        """
        Advanced tire image enhancement with intelligent label protection.
        
        Pipeline:
        1. Detect label regions
        2. Apply CLAHE for local contrast (tread detail)
        3. Lift shadows
        4. Color correction
        5. Denoise
        6. Sharpen (only tire, not label)
        7. Blend: enhanced tire + fully protected labels
        """
        has_alpha = img.mode == 'RGBA'
        
        # Separate alpha channel
        if has_alpha:
            r, g, b, a = img.split()
            rgb_img = Image.merge('RGB', (r, g, b))
            alpha_array = np.array(a)
        else:
            rgb_img = img.convert('RGB')
            alpha_array = None
        
        original_array = np.array(rgb_img, dtype=np.uint8)
        
        # Step 1: Detect label regions (pass alpha for better detection)
        label_mask = self.detect_label_regions(original_array, alpha_array)
        tire_mask = 1.0 - label_mask
        
        # Step 2: Create enhanced version for tire areas
        enhanced = original_array.copy()
        
        # 2a. CLAHE for local contrast (only on tire)
        if self.tire_clahe_clip > 0:
            clahe_result = self.apply_clahe(enhanced, self.tire_clahe_clip, self.tire_clahe_grid)
            # Apply CLAHE only to tire areas
            tire_mask_3ch = np.dstack([tire_mask, tire_mask, tire_mask])
            enhanced = (clahe_result.astype(np.float32) * tire_mask_3ch + 
                       enhanced.astype(np.float32) * (1 - tire_mask_3ch))
            enhanced = enhanced.astype(np.uint8)
        
        # 2b. Shadow lifting (only on tire - dark areas)
        if self.shadow_lift > 0:
            shadow_result = self.lift_shadows(enhanced, self.shadow_lift)
            tire_mask_3ch = np.dstack([tire_mask, tire_mask, tire_mask])
            enhanced = (shadow_result.astype(np.float32) * tire_mask_3ch + 
                       enhanced.astype(np.float32) * (1 - tire_mask_3ch))
            enhanced = enhanced.astype(np.uint8)
        
        # 2c. Color correction (apply to whole image - subtle)
        if self.color_correction:
            enhanced = self.auto_white_balance(enhanced)
        
        # 2d. Denoise (whole image)
        if self.denoise_strength > 0:
            enhanced = self.denoise(enhanced, self.denoise_strength)
        
        # 2e. Brightness/contrast adjustment (only on tire)
        if self.tire_brightness != 1.0 or self.tire_contrast != 1.0:
            enhanced_pil = Image.fromarray(enhanced)
            adjusted = enhanced_pil.copy()
            if self.tire_brightness != 1.0:
                adjusted = ImageEnhance.Brightness(adjusted).enhance(self.tire_brightness)
            if self.tire_contrast != 1.0:
                adjusted = ImageEnhance.Contrast(adjusted).enhance(self.tire_contrast)
            adjusted_array = np.array(adjusted)
            
            # Blend - only apply adjustments to tire
            tire_mask_3ch = np.dstack([tire_mask, tire_mask, tire_mask])
            enhanced = (adjusted_array.astype(np.float32) * tire_mask_3ch + 
                       enhanced.astype(np.float32) * (1 - tire_mask_3ch))
            enhanced = enhanced.astype(np.uint8)
        
        # Step 3: Final blend - use original for label areas with protection level
        label_mask_3ch = np.dstack([label_mask, label_mask, label_mask])
        protection = self.label_protection
        
        # Full protection means 100% original in label areas
        blended = (
            enhanced.astype(np.float32) * (1 - label_mask_3ch * protection) +
            original_array.astype(np.float32) * label_mask_3ch * protection
        )
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Step 4: Sharpening - apply only to tire, not label
        if self.sharpness_factor != 1.0:
            sharpened_pil = Image.fromarray(blended)
            sharpened_pil = ImageEnhance.Sharpness(sharpened_pil).enhance(self.sharpness_factor)
            sharpened_array = np.array(sharpened_pil)
            
            # Blend sharpening - don't sharpen labels
            blended = (sharpened_array.astype(np.float32) * (1 - label_mask_3ch) + 
                      blended.astype(np.float32) * label_mask_3ch)
            blended = blended.astype(np.uint8)
        
        final = Image.fromarray(blended)
        
        # Composite onto white background
        if has_alpha and alpha_array is not None:
            white_bg = Image.new('RGB', final.size, (255, 255, 255))
            alpha_mask = Image.fromarray(alpha_array)
            result = Image.composite(final, white_bg, alpha_mask)
            return result
        
        return final
    
    # ==================== CROPPING ====================
    
    def crop_to_square(self, image_data):
        """Crop to 1:1 with subject centered and margin."""
        if isinstance(image_data, bytes):
            img = Image.open(BytesIO(image_data))
        else:
            img = image_data
        
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        logger.info(f"crop_to_square: input size {img.size}, mode {img.mode}")
        
        # Apply edge refinement if enabled
        if self.edge_refinement:
            img = self.refine_edges(img)
        
        # Get alpha channel for bounding box
        alpha = np.array(img.split()[3])
        
        # Find bounding box using alpha threshold (more robust than getbbox)
        # getbbox can be affected by semi-transparent pixels
        alpha_threshold = 10  # Pixels with alpha > 10 are considered content
        content_mask = alpha > alpha_threshold
        
        if not np.any(content_mask):
            raise Exception("No content found in image")
        
        # Find rows and columns with content
        rows_with_content = np.any(content_mask, axis=1)
        cols_with_content = np.any(content_mask, axis=0)
        
        # Get bounding box
        row_indices = np.where(rows_with_content)[0]
        col_indices = np.where(cols_with_content)[0]
        
        top = row_indices[0]
        bottom = row_indices[-1] + 1
        left = col_indices[0]
        right = col_indices[-1] + 1
        
        bbox = (left, top, right, bottom)
        logger.info(f"crop_to_square: calculated bbox {bbox}")
        
        img_cropped = img.crop(bbox)
        width, height = img_cropped.size
        
        logger.info(f"crop_to_square: cropped content size {width}x{height}")
        
        max_dim = max(width, height)
        margin = int(max_dim * (self.margin_percent / 100))
        square_size = max_dim + (margin * 2)
        
        logger.info(f"crop_to_square: square_size={square_size}, margin={margin}px")
        
        # Create new canvas and center
        square_img = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))
        x_offset = (square_size - width) // 2
        y_offset = (square_size - height) // 2
        
        logger.info(f"crop_to_square: centering offsets x={x_offset}, y={y_offset}")
        
        square_img.paste(img_cropped, (x_offset, y_offset), img_cropped)
        
        # Resize if needed
        if square_size > self.output_size:
            square_img = square_img.resize((self.output_size, self.output_size), Image.Resampling.LANCZOS)
            logger.info(f"crop_to_square: resized to {self.output_size}x{self.output_size}")
        elif square_size < self.output_size:
            # Also resize up if smaller than target
            square_img = square_img.resize((self.output_size, self.output_size), Image.Resampling.LANCZOS)
            logger.info(f"crop_to_square: upscaled to {self.output_size}x{self.output_size}")
        
        return square_img
    
    # ==================== MAIN PIPELINE ====================
    
    def process(self, image_data, bg_method=None):
        """Full processing pipeline."""
        # Step 1: Remove background
        no_bg_image = self.remove_background(image_data, method=bg_method)
        
        # Step 2: Crop to square
        square_img = self.crop_to_square(no_bg_image)
        
        # Step 3: Enhance
        enhanced_img = self.enhance_image(square_img)
        
        # Step 4: Save as high-quality JPEG
        output = BytesIO()
        enhanced_img.save(output, format='JPEG', quality=98, subsampling=0)
        output.seek(0)
        
        return output.getvalue()
    
    def process_with_settings(self, image_data, settings):
        """Process with custom settings (for lab testing)."""
        # Apply all settings
        self.tire_clahe_clip = settings.get('clahe_clip', 2.0)
        self.tire_clahe_grid = settings.get('clahe_grid', 8)
        self.tire_brightness = settings.get('brightness', 1.05)
        self.tire_contrast = settings.get('contrast', 1.10)
        self.shadow_lift = settings.get('shadow_lift', 15)
        
        self.label_threshold = settings.get('label_threshold', 200)
        self.label_protection = settings.get('label_protection', 0.85)
        
        self.sharpness_factor = settings.get('sharpness', 1.15)
        self.color_correction = settings.get('color_correction', True)
        self.denoise_strength = settings.get('denoise', 3)
        
        self.margin_percent = settings.get('margin_percent', 5)
        self.output_size = settings.get('output_size', 2048)
        
        # Background removal settings
        self.alpha_matting = settings.get('alpha_matting', False)
        self.erode_size = settings.get('erode_size', 10)
        self.fg_threshold = settings.get('fg_threshold', 240)
        self.bg_threshold = settings.get('bg_threshold', 10)
        
        # Edge refinement settings
        self.edge_refinement = settings.get('edge_refinement', True)
        self.edge_feather = settings.get('edge_feather', 2)
        self.edge_smooth = settings.get('edge_smooth', 3)
        self.edge_erode = settings.get('edge_erode', 1)
        self.anti_alias = settings.get('anti_alias', True)
        
        remove_bg = settings.get('remove_bg', True)
        bg_method = settings.get('bg_method', 'auto')
        
        # Process
        if remove_bg:
            no_bg_data = self.remove_background(image_data, method=bg_method)
            img = Image.open(BytesIO(no_bg_data))
        else:
            img = Image.open(BytesIO(image_data))
        
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        square_img = self.crop_to_square(img)
        enhanced_img = self.enhance_image(square_img)
        
        output = BytesIO()
        enhanced_img.save(output, format='JPEG', quality=98, subsampling=0)
        output.seek(0)
        
        return output.getvalue()
    
    # ==================== PRESETS ====================
    
    @classmethod
    def preset_default(cls, poof_api_key=None):
        """Balanced preset for most tires."""
        p = cls(poof_api_key=poof_api_key)
        p.tire_clahe_clip = 2.0
        p.tire_brightness = 1.05
        p.tire_contrast = 1.10
        p.shadow_lift = 15
        p.label_protection = 0.85
        p.sharpness_factor = 1.15
        return p
    
    @classmethod
    def preset_high_contrast(cls, poof_api_key=None):
        """For tires with deep tread patterns - more aggressive contrast."""
        p = cls(poof_api_key=poof_api_key)
        p.tire_clahe_clip = 3.0
        p.tire_brightness = 1.08
        p.tire_contrast = 1.20
        p.shadow_lift = 25
        p.label_protection = 0.90
        p.sharpness_factor = 1.25
        return p
    
    @classmethod
    def preset_label_focus(cls, poof_api_key=None):
        """For tires where label readability is critical."""
        p = cls(poof_api_key=poof_api_key)
        p.tire_clahe_clip = 1.5
        p.tire_brightness = 1.02
        p.tire_contrast = 1.05
        p.shadow_lift = 10
        p.label_protection = 0.95
        p.label_threshold = 180
        p.sharpness_factor = 1.10
        return p
    
    @classmethod 
    def preset_natural(cls, poof_api_key=None):
        """Minimal enhancement - preserves original appearance."""
        p = cls(poof_api_key=poof_api_key)
        p.tire_clahe_clip = 1.0
        p.tire_brightness = 1.0
        p.tire_contrast = 1.05
        p.shadow_lift = 10
        p.label_protection = 0.80
        p.sharpness_factor = 1.05
        p.color_correction = False
        return p


# Backwards compatibility - alias to original class name
ImageProcessor = ImageProcessorV2
