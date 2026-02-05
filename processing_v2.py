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
                # Validate response is actually an image (PNG starts with specific bytes)
                content = response.content
                if len(content) < 1000:
                    logger.error(f"Poof API returned too little data: {len(content)} bytes - {content[:200]}")
                    raise Exception(f"Poof API returned invalid data: {content[:200]}")
                if not (content[:4] == b'\x89PNG' or content[:2] == b'\xff\xd8'):
                    logger.error(f"Poof API returned non-image data: {content[:100]}")
                    raise Exception(f"Poof API returned non-image: {content[:100]}")
                logger.info(f"Poof API success - {len(content)} bytes")
                return content
            else:
                logger.error(f"Poof API error: {response.status_code} - {response.text[:200]}")
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
    
    def detect_label_regions(self, img_array, alpha_mask=None, max_labels=1):
        """
        Detect label/text regions on tire.
        Labels are typically white/light rectangles with dark text on the tire.
        
        Args:
            img_array: RGB image array
            alpha_mask: Optional alpha channel to identify tire vs background
            max_labels: Maximum number of labels to detect (default 1)
            
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
        
        # Pre-compute: which bright regions connect to the image border?
        # These are backdrop (through tire hole), not labels on the tire.
        num_cc, cc_labels = cv2.connectedComponents(bright_mask, connectivity=8)
        border_cc = set()
        border_cc.update(cc_labels[0, :].tolist())       # top
        border_cc.update(cc_labels[-1, :].tolist())       # bottom
        border_cc.update(cc_labels[:, 0].tolist())        # left
        border_cc.update(cc_labels[:, -1].tolist())       # right
        border_cc.discard(0)
        
        # Create border-connected mask for quick lookup
        border_connected = np.zeros((h, w), dtype=bool)
        for lbl in border_cc:
            border_connected[cc_labels == lbl] = True
        
        # Collect candidate labels with their scores
        label_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size - labels are typically 1-15% of tire area
            tire_area = np.sum(tire_region)
            min_area = tire_area * 0.002  # Min 0.2% of tire
            max_area = tire_area * 0.15   # Max 15% of tire
            
            if area < min_area or area > max_area:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Check aspect ratio - labels are usually rectangular
            aspect = max(cw, ch) / (min(cw, ch) + 1)
            if aspect > 8:  # Too elongated
                continue
            
            # Reject if this region overlaps with border-connected bright areas
            # (backdrop visible through tire hole, not a label)
            roi_border = border_connected[y:y+ch, x:x+cw]
            if np.mean(roi_border) > 0.3:  # >30% of region touches border-connected area
                logger.debug(f"Rejected border-connected region: ({x},{y}) {cw}x{ch}")
                continue
            
            # Reject if shape doesn't fill bounding box well (irregular = backdrop through tire hole)
            # Real labels are rectangular paper, so fill ratio > 0.5
            rect_fill = area / (cw * ch) if cw * ch > 0 else 0
            if rect_fill < 0.45:
                logger.debug(f"Rejected low-fill region: ({x},{y}) {cw}x{ch}, fill={rect_fill:.2f}")
                continue
            
            # Check if region has text-like contrast (dark on light)
            roi = gray[y:y+ch, x:x+cw]
            roi_std = np.std(roi)
            
            # Labels have high local contrast (text)
            if roi_std > 20:  # Has dark text on light background
                # Score by area (prefer larger labels)
                label_candidates.append({
                    'x': x, 'y': y, 'w': cw, 'h': ch,
                    'area': area, 'std': roi_std,
                    'score': area * (roi_std / 50)  # Score considers size and contrast
                })
        
        # Sort by score (descending) and keep only max_labels
        label_candidates.sort(key=lambda c: c['score'], reverse=True)
        label_candidates = label_candidates[:max_labels]
        
        # Mark selected labels
        for label in label_candidates:
            x, y, cw, ch = label['x'], label['y'], label['w'], label['h']
            pad = 15
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + cw + pad), min(h, y + ch + pad)
            label_mask[y1:y2, x1:x2] = 1.0
            logger.debug(f"Label detected: ({x},{y}) {cw}x{ch}, std={label['std']:.1f}")
        
        # 3. Also protect small pure white areas on the tire (label paper, stickers)
        # But NOT large white areas (backdrop visible through tire hole)
        # Skip this if we already have max labels
        if len(label_candidates) < max_labels:
            pure_white_on_tire = (brightness > 230) & tire_region
            pure_white_mask = pure_white_on_tire.astype(np.uint8) * 255
            
            # Only keep small white regions (actual labels), reject large ones (backdrop)
            num_wlabels, wlabels, wstats, _ = cv2.connectedComponentsWithStats(pure_white_mask, connectivity=8)
            filtered_white = np.zeros((h, w), dtype=np.float32)
            for i in range(1, num_wlabels):
                area = wstats[i, cv2.CC_STAT_AREA]
                # Labels are small (< 5% of image), backdrop is large
                if area < h * w * 0.05:
                    filtered_white[wlabels == i] = 1.0
            
            label_mask = np.maximum(label_mask, filtered_white)
        
        # 4. Smooth the mask edges for blending
        if np.any(label_mask > 0):
            label_mask = cv2.GaussianBlur(label_mask, (15, 15), 0)
            label_mask = np.clip(label_mask, 0, 1)
        
        coverage = np.mean(label_mask) * 100
        logger.info(f"Label detection: {len(label_candidates)} label(s), {coverage:.1f}% of image protected")
        
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
    
    def enhance_image(self, img, skip_label=False):
        """
        Advanced tire image enhancement with intelligent label protection.
        
        Pipeline:
        1. Detect label regions (skipped for side images)
        2. Apply CLAHE for local contrast (tread detail)
        3. Lift shadows
        4. Color correction
        5. Denoise
        6. Sharpen (only tire, not label)
        7. Blend: enhanced tire + fully protected labels
        
        Args:
            img: PIL Image
            skip_label: If True, skip label detection (for side images)
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
        if skip_label:
            logger.info("Skipping label detection (side image)")
            label_mask = np.zeros((original_array.shape[0], original_array.shape[1]), dtype=np.float32)
        else:
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
    
    def process(self, image_data, bg_method=None, image_type='front'):
        """Full processing pipeline.
        
        Args:
            image_data: Raw image bytes
            bg_method: Background removal method
            image_type: 'front' (detect 1 label), 'side' (no label detection), 'gauge'
        """
        # Step 1: Remove background
        no_bg_image = self.remove_background(image_data, method=bg_method)
        
        # Step 2: Crop to square
        square_img = self.crop_to_square(no_bg_image)
        
        # Step 3: Enhance (skip label detection for side images)
        skip_label = (image_type == 'side')
        enhanced_img = self.enhance_image(square_img, skip_label=skip_label)
        
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
    # ==================== GAUGE DETECTION ====================
    
    def detect_gauge(self, img_array):
        """
        Detect the tread depth gauge in a tire photo.
        Returns (cx, cy, confidence) or None.
        
        Uses two signals:
        1. LCD display: small bright rectangle with high contrast to dark surroundings
        2. Blue gauge tip: distinctive saturated blue color against dark tire
        """
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        candidates = []
        
        # === Signal 1: LCD Display ===
        search_h = int(h * 0.65)
        upper = gray[:search_h, :]
        
        thresh = cv2.adaptiveThreshold(upper, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, -10)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < 150 or area > 8000:
                continue
            
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = cw / (ch + 1)
            
            if aspect < 1.0 or aspect > 6:
                continue
            if cw < 30 or cw > 200 or ch < 15 or ch > 100:
                continue
            
            inside_mean = np.mean(upper[y:y+ch, x:x+cw])
            
            pad = 25
            sy1, sy2 = max(0, y - pad), min(search_h, y + ch + pad)
            sx1, sx2 = max(0, x - pad), min(w, x + cw + pad)
            
            surround = upper[sy1:sy2, sx1:sx2].copy()
            mask = np.zeros_like(surround, dtype=bool)
            mask[y-sy1:y-sy1+ch, x-sx1:x-sx1+cw] = True
            border_mean = np.mean(surround[~mask]) if np.sum(~mask) > 0 else 0
            
            contrast = inside_mean - border_mean
            
            # LCD: brighter than surroundings, dark surroundings (gauge body)
            if contrast > 10 and inside_mean > 40 and border_mean < 130:
                # Center bias: gauge is on tire which is roughly centered
                cx_norm = (x + cw // 2) / w  # 0=left, 0.5=center, 1=right
                center_dist = abs(cx_norm - 0.5) * 2  # 0=center, 1=edge
                center_score = 1.0 - center_dist * 0.5  # Penalize off-center
                
                candidates.append({
                    'type': 'lcd',
                    'cx': x + cw // 2, 'cy': y + ch // 2,
                    'score': contrast * (1.5 if 1.5 < aspect < 4 else 0.5) * center_score
                })
        
        # === Signal 2: Blue gauge tip (wider range for different lighting) ===
        blue_mask = cv2.inRange(hsv[:search_h, :], (90, 40, 25), (135, 255, 255))
        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in blue_contours:
            area = cv2.contourArea(c)
            if 20 < area < 3000:
                x, y, cw, ch = cv2.boundingRect(c)
                candidates.append({
                    'type': 'blue_tip',
                    'cx': x + cw // 2, 'cy': y + ch // 2,
                    'score': min(area, 500)
                })
        
        # === Signal 3: Mechanical dial gauge (circular, bright face) ===
        # Always run - dial gauge may be present even when blue tips are found on tread
        blurred = cv2.GaussianBlur(gray[:search_h, :], (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                    param1=100, param2=40, minRadius=15, maxRadius=100)
        if circles is not None:
            circles_arr = np.round(circles[0]).astype(int)
            for cx, cy, r in circles_arr:
                # Dial face should be bright with low saturation
                mask = np.zeros(gray[:search_h, :].shape, np.uint8)
                cv2.circle(mask, (cx, cy), r, 255, -1)
                mean_bright = np.mean(gray[:search_h, :][mask > 0])
                mean_sat = np.mean(hsv[:search_h, :, 1][mask > 0])
                
                if mean_bright > 150 and mean_sat < 80:
                    # Center bias
                    cx_norm = cx / w
                    center_dist = abs(cx_norm - 0.5) * 2
                    center_score = 1.0 - center_dist * 0.5
                    
                    # Must be in central area horizontally
                    if cx < w * 0.2 or cx > w * 0.8:
                        continue
                    
                    # Dial gauge sits on top of tire tread - must be in upper 40%
                    if cy > h * 0.40:
                        continue
                    
                    candidates.append({
                        'type': 'dial',
                        'cx': cx, 'cy': cy,
                        'radius': r,
                        'score': mean_bright * center_score * 0.5
                    })
                    logger.debug(f"Dial candidate: ({cx},{cy}) r={r}, bright={mean_bright:.0f}, sat={mean_sat:.0f}")
        
        if not candidates:
            return None
        
        # Filter: candidates should be in the central-upper area of the image
        filtered = []
        for c in candidates:
            if c['type'] == 'lcd':
                if c['cx'] < w * 0.15 or c['cx'] > w * 0.85:
                    continue
                if c['cy'] > h * 0.50:
                    continue
            if c['type'] == 'blue_tip':
                if c['cx'] < w * 0.2 or c['cx'] > w * 0.8:
                    continue
            if c['type'] == 'dial':
                # Dial already filtered during detection
                pass
            filtered.append(c)
        
        if not filtered:
            return None
        
        candidates = filtered
        
        # Combine signals: LCD + blue tip (digital) OR dial (mechanical)
        lcds = [c for c in candidates if c['type'] == 'lcd']
        tips = [c for c in candidates if c['type'] == 'blue_tip']
        dials = [c for c in candidates if c['type'] == 'dial']
        
        # Try digital gauge first (LCD candidates)
        if lcds:
            # Filter out clusters of LCD candidates that look like label text rows
            if len(lcds) >= 2:
                lcds_sorted = sorted(lcds, key=lambda c: c['cy'])
                y_spread = lcds_sorted[-1]['cy'] - lcds_sorted[0]['cy']
                if y_spread > 120:
                    logger.info(f"LCD candidates spread {y_spread}px vertically - likely label text")
                    lcds = []  # Reject LCD, may still have dial
            
            if lcds:
                best_score = 0
                best_pos = None
                for lcd in lcds:
                    score = lcd['score']
                    for tip in tips:
                        dist = ((lcd['cx'] - tip['cx'])**2 + (lcd['cy'] - tip['cy'])**2)**0.5
                        if dist < 250:
                            score += tip['score'] * 2
                    if score > best_score:
                        best_score = score
                        best_pos = (lcd['cx'], lcd['cy'])
                
                if best_pos:
                    confidence = min(best_score / 100, 1.0)
                    logger.info(f"Digital gauge at ({best_pos[0]},{best_pos[1]}), confidence={confidence:.2f}")
                    return (best_pos[0], best_pos[1], confidence, 'digital')
        
        # Try mechanical dial gauge
        if dials:
            best_dial = max(dials, key=lambda c: c['score'])
            confidence = min(best_dial['score'] / 100, 1.0)
            logger.info(f"Dial gauge at ({best_dial['cx']},{best_dial['cy']}), r={best_dial.get('radius',0)}, confidence={confidence:.2f}")
            return (best_dial['cx'], best_dial['cy'], confidence, 'dial', best_dial.get('radius', 40))
        
        return None
    
    def crop_gauge_closeup(self, image_data, gauge_xy=None, output_size=None):
        """
        Crop around the tread depth gauge for a close-up measurement image.
        
        Pipeline:
        1. Detect or use provided gauge position
        2. Crop tightly around gauge + hand + tire tread (exclude label)
        3. Place on clean white background, centered
        4. Enhance for readability
        
        No background removal — the hand holding the gauge must stay visible.
        
        Args:
            image_data: Raw image bytes or PIL Image
            gauge_xy: Optional (x, y) tuple. If None, auto-detect.
            output_size: Output image size (default: self.output_size)
            
        Returns:
            (jpeg_bytes, message) or (None, error_message)
        """
        output_size = output_size or self.output_size
        
        if isinstance(image_data, bytes):
            img = Image.open(BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, Image.Image):
            img = image_data if image_data.mode == 'RGB' else image_data.convert('RGB')
        else:
            img = image_data
        
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        if gauge_xy is None:
            result = self.detect_gauge(img_array)
            if result is None:
                return None, "No gauge detected in image"
            cx, cy, conf = result[0], result[1], result[2]
            gauge_type = result[3] if len(result) > 3 else 'digital'
            dial_radius = result[4] if len(result) > 4 else 40
            if conf < 0.5:
                return None, f"Low confidence gauge detection ({conf:.2f})"
        else:
            cx, cy = gauge_xy
            conf = 1.0
            dial_radius = 40  # default
            
            # Try to detect dial circle near click point (mechanical gauge)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            search_r = int(min(w, h) * 0.15)
            sy1 = max(0, cy - search_r)
            sy2 = min(h, cy + search_r)
            sx1 = max(0, cx - search_r)
            sx2 = min(w, cx + search_r)
            roi = gray[sy1:sy2, sx1:sx2]
            
            blurred = cv2.GaussianBlur(roi, (9, 9), 2)
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                        param1=100, param2=30, minRadius=15, maxRadius=int(search_r*0.8))
            
            gauge_type = 'manual'
            if circles is not None:
                # Find the circle closest to click point (center of ROI)
                roi_cx, roi_cy = cx - sx1, cy - sy1
                best_dist = float('inf')
                for c in circles[0]:
                    dcx, dcy, dr = int(c[0]), int(c[1]), int(c[2])
                    # Check brightness (dial face is white)
                    mask = np.zeros(roi.shape, np.uint8)
                    cv2.circle(mask, (dcx, dcy), dr, 255, -1)
                    mean_bright = np.mean(roi[mask > 0])
                    mean_sat = np.mean(hsv[sy1:sy2, sx1:sx2, 1][mask > 0])
                    if mean_bright > 140 and mean_sat < 90:
                        dist = ((dcx - roi_cx)**2 + (dcy - roi_cy)**2)**0.5
                        if dist < best_dist:
                            best_dist = dist
                            dial_radius = dr
                            # Snap to circle center
                            cx = sx1 + dcx
                            cy = sy1 + dcy
                            gauge_type = 'dial'
                
                if gauge_type == 'dial':
                    logger.info(f"Manual click snapped to dial: ({cx},{cy}) r={dial_radius}")
        
        # Step 1: Tight crop around gauge
        min_dim = min(w, h)
        
        if gauge_type == 'dial':
            # Dial gauge: use detected radius to size crop so dial is readable
            # We want the dial to be ~25-30% of the crop width
            target_dial_pct = 0.28  # Dial should be 28% of crop width
            ideal_side = int(2 * dial_radius / target_dial_pct)
            # Clamp to reasonable range
            ideal_side = max(ideal_side, int(min_dim * 0.12))
            ideal_side = min(ideal_side, int(min_dim * 0.30))
            
            crop_radius_x = ideal_side // 2
            crop_radius_up = int(dial_radius * 2.0)   # Just enough above dial cap
            crop_radius_down = int(ideal_side * 0.7)   # More below for tread
        else:
            # Digital gauge or manual: hand present, LCD center
            crop_radius_x = int(min_dim * 0.18)
            crop_radius_up = int(min_dim * 0.20)   # More room above (hand)
            crop_radius_down = int(min_dim * 0.10)  # Less below (avoid label)
        
        x1 = max(0, cx - crop_radius_x)
        y1 = max(0, cy - crop_radius_up)
        x2 = min(w, cx + crop_radius_x)
        y2 = min(h, cy + crop_radius_down)
        
        # Make square
        crop_w = x2 - x1
        crop_h = y2 - y1
        side = max(crop_w, crop_h)
        
        # Re-center keeping the bias
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        x1 = max(0, center_x - side // 2)
        y1 = max(0, center_y - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        if x2 - x1 < side:
            x1 = max(0, x2 - side)
        if y2 - y1 < side:
            y1 = max(0, y2 - side)
        
        crop = img_array[y1:y2, x1:x2]
        crop_img = Image.fromarray(crop)
        
        logger.info(f"Gauge crop: ({x1},{y1})-({x2},{y2}), size={crop_img.size}")
        
        # Step 2: Whiten the backdrop using edge-connected component analysis
        # Only whitens bright pixels connected to image borders (backdrop fabric),
        # preserving bright areas enclosed by dark pixels (LCD screen, gauge markings)
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        crop_h, crop_w = gray_crop.shape
        
        # Find all bright pixel regions (potential backdrop)
        potential_bg = (gray_crop > 150).astype(np.uint8) * 255
        
        # For dial gauges: detect the exact dial face circle in the crop
        # We'll restore it AFTER whitening instead of blocking whitening
        dial_restore_mask = None
        if gauge_type == 'dial':
            dial_cx_crop = cx - x1
            dial_cy_crop = cy - y1
            # Find the precise dial circle in the crop using HoughCircles
            blurred_crop = cv2.GaussianBlur(gray_crop, (9, 9), 2)
            dial_circles = cv2.HoughCircles(blurred_crop, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                             param1=100, param2=30, minRadius=15, maxRadius=int(min(crop_h, crop_w)*0.2))
            if dial_circles is not None:
                # Find the circle closest to expected dial position
                best_circle = None
                best_dist = float('inf')
                for dc in dial_circles[0]:
                    dcx, dcy, dr = int(dc[0]), int(dc[1]), int(dc[2])
                    dist = ((dcx - dial_cx_crop)**2 + (dcy - dial_cy_crop)**2)**0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_circle = (dcx, dcy, dr)
                
                if best_circle and best_dist < 100:
                    dcx, dcy, dr = best_circle
                    # Create restore mask: tight circle around actual dial face + small margin
                    # Plus the gauge body/stem below the dial
                    dial_restore_mask = np.zeros((crop_h, crop_w), np.uint8)
                    cv2.circle(dial_restore_mask, (dcx, dcy), dr + 8, 255, -1)
                    # Add stem area below dial (rectangular, extends down into tire)
                    stem_w = max(dr // 2, 15)
                    cv2.rectangle(dial_restore_mask, 
                                  (dcx - stem_w, dcy + dr - 5), 
                                  (dcx + stem_w, min(crop_h, dcy + dr + int(dr * 2.5))), 255, -1)
                    logger.debug(f"Dial restore: circle ({dcx},{dcy}) r={dr+8}, stem below")
        
        # Label connected components
        num_labels, labels = cv2.connectedComponents(potential_bg, connectivity=8)
        
        # Find which component labels touch any border
        border_labels = set()
        border_labels.update(labels[0, :].tolist())       # top
        border_labels.update(labels[-1, :].tolist())       # bottom
        border_labels.update(labels[:, 0].tolist())        # left
        border_labels.update(labels[:, -1].tolist())       # right
        border_labels.discard(0)  # 0 = background (dark pixels)
        
        # Create mask of only border-connected bright regions
        bg_binary = np.zeros((crop_h, crop_w), np.uint8)
        for label in border_labels:
            bg_binary[labels == label] = 255
        
        # Dilate slightly to catch edge fringing
        bg_binary = cv2.dilate(bg_binary, np.ones((5, 5), np.uint8), iterations=1)
        
        # Create smooth float mask
        bg_float = cv2.GaussianBlur(bg_binary.astype(np.float32) / 255, (31, 31), 0)
        
        # Residual: gently push very bright low-saturation pixels not reached by flood
        # (catches small disconnected fabric patches between fingers etc.)
        sat = hsv_crop[:,:,1].astype(np.float32)
        residual_bright = np.clip((gray_crop.astype(np.float32) - 185) / 35, 0, 1)
        residual_sat = np.clip((25 - sat) / 20, 0, 1)
        residual = residual_bright * residual_sat * (1 - bg_float) * 0.7
        residual = cv2.GaussianBlur(residual, (15, 15), 0)
        
        combined_mask = np.clip(bg_float + residual, 0, 1)
        
        # Blend original with white using mask
        crop_f = crop.astype(np.float32)
        white = np.full_like(crop_f, 255.0)
        whitened = crop_f * (1 - combined_mask[:,:,np.newaxis]) + white * combined_mask[:,:,np.newaxis]
        whitened = np.clip(whitened, 0, 255).astype(np.uint8)
        
        # For dial gauges: restore the dial face from original over the whitened result
        if dial_restore_mask is not None:
            restore_float = cv2.GaussianBlur(dial_restore_mask.astype(np.float32) / 255, (11, 11), 0)
            restore_3ch = restore_float[:,:,np.newaxis]
            whitened = (crop.astype(np.float32) * restore_3ch + 
                       whitened.astype(np.float32) * (1 - restore_3ch))
            whitened = np.clip(whitened, 0, 255).astype(np.uint8)
        
        crop_img = Image.fromarray(whitened)
        logger.info(f"Backdrop whitening: {len(border_labels)} edge regions, {np.mean(bg_binary > 0)*100:.0f}% coverage")
        
        # Step 3: Resize to output size
        crop_img = crop_img.resize((output_size, output_size), Image.Resampling.LANCZOS)
        
        # Step 3: Enhance for readability
        crop_img = ImageEnhance.Sharpness(crop_img).enhance(1.3)
        crop_img = ImageEnhance.Contrast(crop_img).enhance(1.05)
        
        # Save to bytes
        output = BytesIO()
        crop_img.save(output, format='JPEG', quality=98, subsampling=0)
        output.seek(0)
        
        msg = f"Gauge closeup ({gauge_type}) from ({cx},{cy}), confidence={conf:.2f}"
        logger.info(msg)
        return output.getvalue(), msg


ImageProcessor = ImageProcessorV2
