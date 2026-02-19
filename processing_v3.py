"""
SmartPneu Image Processing v3
Clean, professional tire photos for e-commerce.

Philosophy: Less is more. The goal is a clean studio look, not heavy processing.

Pipeline:
1. Background removal (Poof API with rembg fallback)
2. Light edge cleanup (remove fringe, no aggressive erosion)
3. Smart square crop (centered, consistent margins)
4. Minimal enhancement (slight contrast + sharpness only)
5. Composite on clean white background
6. High-quality JPEG output

Image types:
- front: Tire face with SmartPneu label visible. Detect & protect label.
- side: Tire sidewall showing brand/size markings. No label detection, slightly stronger enhancement.
- gauge: Handled separately (crop_gauge_closeup, unchanged from v2)
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


class ImageProcessorV3:
    """Clean tire image processor focused on professional e-commerce output."""
    
    def __init__(self, poof_api_key=None, output_size=2048):
        self.output_size = output_size
        self.poof_api_key = poof_api_key or os.environ.get('POOF_API_KEY')
        
        # ===== ENHANCEMENT (kept minimal) =====
        # Front images (with label)
        self.front_contrast = 1.08        # Slight contrast boost
        self.front_sharpness = 1.10       # Slight sharpening
        self.front_brightness = 1.0       # No brightness change
        
        # Side images (no label, show tread/sidewall detail)
        self.side_contrast = 1.12         # Slightly more contrast
        self.side_sharpness = 1.15        # Slightly more sharpening  
        self.side_brightness = 1.03       # Very subtle brightening
        
        # Label protection
        self.label_threshold = 200        # Brightness threshold for label detection
        self.label_protection = 1.0       # Full protection
        
        # Crop settings
        self.margin_percent = 4           # Margin around tire in output
        
        # Background removal
        self.bg_removal_method = 'poof' if self.poof_api_key else 'rembg'
        self.alpha_matting = False
        self.erode_size = 10
        self.fg_threshold = 240
        self.bg_threshold = 10
        
        # Edge cleanup (lighter than v2)
        self.edge_erode = 1               # Just 1 iteration (was 3 in v2)
        self.edge_smooth = 2              # Light smoothing
        self.edge_feather = 1             # Minimal feather
    
    # ==================== BACKGROUND REMOVAL ====================
    # (Kept identical to v2 — Poof API with rembg fallback)
    
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
                content = response.content
                if len(content) < 1000:
                    logger.error(f"Poof API returned too little data: {len(content)} bytes")
                    raise Exception(f"Poof API returned invalid data: {content[:200]}")
                if not (content[:4] == b'\x89PNG' or content[:2] == b'\xff\xd8'):
                    logger.error(f"Poof API returned non-image data")
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
    
    # ==================== EDGE CLEANUP ====================
    
    def cleanup_edges(self, img):
        """
        Light edge cleanup — remove color fringe without eating into the tire.
        Much lighter touch than v2's refine_edges.
        """
        if img.mode != 'RGBA':
            return img
        
        r, g, b, a = img.split()
        alpha = np.array(a, dtype=np.uint8)
        
        # Step 1: Light erosion to remove 1px of color fringe
        if self.edge_erode > 0:
            kernel = np.ones((3, 3), np.uint8)
            alpha = cv2.erode(alpha, kernel, iterations=self.edge_erode)
        
        # Step 2: Smooth edge transitions
        if self.edge_smooth > 0:
            # Find edge region
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(alpha, kernel, iterations=1)
            eroded = cv2.erode(alpha, kernel, iterations=1)
            edge_band = (dilated - eroded) > 0
            
            # Smooth only the edges
            blur_size = self.edge_smooth * 2 + 1
            alpha_smooth = cv2.GaussianBlur(alpha.astype(np.float32), (blur_size, blur_size), 0)
            edge_float = edge_band.astype(np.float32)
            edge_float = cv2.GaussianBlur(edge_float, (blur_size, blur_size), 0)
            
            alpha = (alpha_smooth * edge_float + alpha.astype(np.float32) * (1 - edge_float))
            alpha = np.clip(alpha, 0, 255).astype(np.uint8)
        
        # Step 3: Ensure core areas stay fully opaque/transparent
        original_alpha = np.array(a, dtype=np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        
        core_opaque = cv2.erode((original_alpha == 255).astype(np.uint8) * 255, kernel, iterations=2)
        alpha = np.where(core_opaque > 0, 255, alpha)
        
        core_transparent = cv2.erode((original_alpha == 0).astype(np.uint8) * 255, kernel, iterations=2)
        alpha = np.where(core_transparent > 0, 0, alpha)
        
        return Image.merge('RGBA', (r, g, b, Image.fromarray(alpha)))
    
    # ==================== LABEL DETECTION ====================
    
    def detect_label_mask(self, img_array, alpha_mask=None):
        """
        Detect the SmartPneu label on the tire.
        Returns a float mask: 1.0 = label area (protect), 0.0 = tire area (enhance).
        
        Simplified from v2: focuses on finding the single rectangular label.
        """
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        h, w = gray.shape
        label_mask = np.zeros((h, w), dtype=np.float32)
        
        # Determine tire region (opaque pixels)
        if alpha_mask is not None:
            tire_region = alpha_mask > 128
        else:
            tire_region = gray < 240
        
        # Find bright regions ON the tire (potential labels)
        brightness = gray.astype(np.float32)
        tire_pixels = brightness[tire_region & (brightness < 200)]
        if len(tire_pixels) == 0:
            return label_mask
        
        tire_avg = np.mean(tire_pixels)
        tire_std = np.std(tire_pixels)
        label_brightness_threshold = min(tire_avg + 3 * tire_std, self.label_threshold)
        
        # Bright-on-tire mask
        bright_on_tire = (brightness > label_brightness_threshold) & tire_region
        bright_mask = bright_on_tire.astype(np.uint8) * 255
        
        # Cleanup
        kernel = np.ones((5, 5), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Filter out border-connected regions (backdrop through tire hole, not labels)
        num_cc, cc_labels = cv2.connectedComponents(bright_mask, connectivity=8)
        border_cc = set()
        border_cc.update(cc_labels[0, :].tolist())
        border_cc.update(cc_labels[-1, :].tolist())
        border_cc.update(cc_labels[:, 0].tolist())
        border_cc.update(cc_labels[:, -1].tolist())
        border_cc.discard(0)
        
        border_connected = np.zeros((h, w), dtype=bool)
        for lbl in border_cc:
            border_connected[cc_labels == lbl] = True
        
        # Find contours and score them
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tire_area = max(np.sum(tire_region), 1)
        
        best_label = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < tire_area * 0.002 or area > tire_area * 0.15:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = max(cw, ch) / (min(cw, ch) + 1)
            if aspect > 8:
                continue
            
            # Reject border-connected (backdrop through hole)
            roi_border = border_connected[y:y+ch, x:x+cw]
            if np.mean(roi_border) > 0.3:
                continue
            
            # Reject irregular shapes (backdrop through hole)
            rect_fill = area / (cw * ch) if cw * ch > 0 else 0
            if rect_fill < 0.45:
                continue
            
            # Score: size × contrast (text on label)
            roi = gray[y:y+ch, x:x+cw]
            roi_std = np.std(roi)
            if roi_std < 20:
                continue
            
            score = area * (roi_std / 50)
            if score > best_score:
                best_score = score
                best_label = (x, y, cw, ch)
        
        # Apply the best label
        if best_label:
            x, y, cw, ch = best_label
            pad = 20
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + cw + pad), min(h, y + ch + pad)
            label_mask[y1:y2, x1:x2] = 1.0
            
            # Smooth edges for blending
            label_mask = cv2.GaussianBlur(label_mask, (15, 15), 0)
            label_mask = np.clip(label_mask, 0, 1)
            
            coverage = np.mean(label_mask) * 100
            logger.info(f"Label detected: ({x},{y}) {cw}x{ch}, protection coverage: {coverage:.1f}%")
        else:
            logger.info("No label detected")
        
        return label_mask
    
    # ==================== ENHANCEMENT ====================
    
    def enhance_tire(self, img, image_type='front'):
        """
        Minimal enhancement for a clean, professional look.
        
        For front: very light contrast + sharpness, protect label
        For side: slightly stronger contrast + sharpness, no label detection
        """
        has_alpha = img.mode == 'RGBA'
        
        # Separate alpha
        if has_alpha:
            r, g, b, a = img.split()
            rgb_img = Image.merge('RGB', (r, g, b))
            alpha_array = np.array(a)
        else:
            rgb_img = img.convert('RGB')
            alpha_array = None
        
        original_array = np.array(rgb_img, dtype=np.uint8)
        
        # Get enhancement parameters
        if image_type == 'side':
            contrast = self.side_contrast
            sharpness = self.side_sharpness
            brightness = self.side_brightness
            label_mask = np.zeros(original_array.shape[:2], dtype=np.float32)
            logger.info(f"Side image enhancement: contrast={contrast}, sharpness={sharpness}")
        else:
            contrast = self.front_contrast
            sharpness = self.front_sharpness
            brightness = self.front_brightness
            label_mask = self.detect_label_mask(original_array, alpha_array)
            logger.info(f"Front image enhancement: contrast={contrast}, sharpness={sharpness}")
        
        tire_mask = 1.0 - label_mask
        
        # Apply enhancement to the whole image
        enhanced_pil = rgb_img.copy()
        
        if brightness != 1.0:
            enhanced_pil = ImageEnhance.Brightness(enhanced_pil).enhance(brightness)
        if contrast != 1.0:
            enhanced_pil = ImageEnhance.Contrast(enhanced_pil).enhance(contrast)
        if sharpness != 1.0:
            enhanced_pil = ImageEnhance.Sharpness(enhanced_pil).enhance(sharpness)
        
        enhanced_array = np.array(enhanced_pil, dtype=np.uint8)
        
        # Blend: enhanced tire + original label
        if np.any(label_mask > 0):
            label_mask_3ch = np.dstack([label_mask, label_mask, label_mask])
            blended = (
                enhanced_array.astype(np.float32) * (1 - label_mask_3ch * self.label_protection) +
                original_array.astype(np.float32) * label_mask_3ch * self.label_protection
            )
            final_array = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            final_array = enhanced_array
        
        final = Image.fromarray(final_array)
        
        # Composite on white background
        if has_alpha and alpha_array is not None:
            bg = Image.new('RGB', final.size, (255, 255, 255))
            alpha_img = Image.fromarray(alpha_array)
            result = Image.composite(final, bg, alpha_img)
            return result
        
        return final
    
    # ==================== CROPPING ====================
    
    def crop_to_square(self, image_data):
        """
        Crop to 1:1 with tire centered and consistent margin.
        Applies edge cleanup before cropping.
        """
        if isinstance(image_data, bytes):
            img = Image.open(BytesIO(image_data))
        else:
            img = image_data
        
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        logger.info(f"crop_to_square: input {img.size}, mode {img.mode}")
        
        # Edge cleanup
        img = self.cleanup_edges(img)
        
        # Find content bounding box from alpha
        alpha = np.array(img.split()[3])
        content_mask = alpha > 10
        
        if not np.any(content_mask):
            raise Exception("No content found in image after background removal")
        
        rows = np.any(content_mask, axis=1)
        cols = np.any(content_mask, axis=0)
        row_idx = np.where(rows)[0]
        col_idx = np.where(cols)[0]
        
        top, bottom = row_idx[0], row_idx[-1] + 1
        left, right = col_idx[0], col_idx[-1] + 1
        
        logger.info(f"crop_to_square: content bbox ({left},{top})-({right},{bottom})")
        
        # Crop to content
        img_cropped = img.crop((left, top, right, bottom))
        cw, ch = img_cropped.size
        
        # Calculate square canvas
        max_dim = max(cw, ch)
        margin = int(max_dim * (self.margin_percent / 100))
        square_size = max_dim + (margin * 2)
        
        # Create canvas and center
        canvas = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))
        x_offset = (square_size - cw) // 2
        y_offset = (square_size - ch) // 2
        canvas.paste(img_cropped, (x_offset, y_offset), img_cropped)
        
        # Resize to output size
        if square_size != self.output_size:
            canvas = canvas.resize((self.output_size, self.output_size), Image.Resampling.LANCZOS)
            logger.info(f"crop_to_square: resized {square_size} -> {self.output_size}")
        
        return canvas
    
    # ==================== MAIN PIPELINE ====================
    
    def process(self, image_data, bg_method=None, image_type='front'):
        """
        Full processing pipeline for front and side images.
        
        Pipeline:
        1. Remove background (Poof/rembg)
        2. Crop to square with edge cleanup
        3. Minimal enhancement (protect label for front)
        4. Composite on white background
        5. Output high-quality JPEG
        
        Args:
            image_data: Raw JPEG bytes from camera
            bg_method: Override bg removal method
            image_type: 'front', 'side', or 'gauge'
        
        Returns:
            JPEG bytes
        """
        logger.info(f"Processing {image_type} image ({len(image_data)} bytes)")
        
        # Step 1: Remove background
        no_bg_data = self.remove_background(image_data, method=bg_method)
        
        # Step 2: Crop to square (includes edge cleanup)
        square_img = self.crop_to_square(no_bg_data)
        
        # Step 3: Enhance + composite on white
        final_img = self.enhance_tire(square_img, image_type=image_type)
        
        # Step 4: Output JPEG
        output = BytesIO()
        final_img.save(output, format='JPEG', quality=98, subsampling=0)
        output.seek(0)
        
        logger.info(f"Processing complete: {self.output_size}x{self.output_size} JPEG")
        return output.getvalue()
    
    def process_with_settings(self, image_data, settings):
        """Process with custom settings (for lab testing).
        
        Accepts both v2 and v3 setting names for backwards compatibility.
        v2-specific settings (clahe_clip, shadow_lift, denoise, etc.) are ignored.
        """
        # Enhancement settings
        self.front_contrast = settings.get('contrast', 1.08)
        self.front_sharpness = settings.get('sharpness', 1.10)
        self.front_brightness = settings.get('brightness', 1.0)
        self.side_contrast = settings.get('side_contrast', settings.get('contrast', 1.12))
        self.side_sharpness = settings.get('side_sharpness', settings.get('sharpness', 1.15))
        self.side_brightness = settings.get('side_brightness', settings.get('brightness', 1.03))
        
        # Label settings
        self.label_threshold = settings.get('label_threshold', 200)
        self.label_protection = settings.get('label_protection', 1.0)
        
        # Crop settings
        self.margin_percent = settings.get('margin_percent', settings.get('margin', 4))
        self.output_size = settings.get('output_size', 2048)
        
        # Background removal settings
        self.alpha_matting = settings.get('alpha_matting', False)
        self.erode_size = settings.get('erode_size', 10)
        self.fg_threshold = settings.get('fg_threshold', 240)
        self.bg_threshold = settings.get('bg_threshold', 10)
        
        # Edge cleanup settings
        self.edge_erode = settings.get('edge_erode', 1)
        self.edge_smooth = settings.get('edge_smooth', 2)
        self.edge_feather = settings.get('edge_feather', 1)
        
        remove_bg = settings.get('remove_bg', True)
        bg_method = settings.get('bg_method', 'auto')
        image_type = settings.get('image_type', 'front')
        
        if remove_bg:
            no_bg_data = self.remove_background(image_data, method=bg_method)
            img = Image.open(BytesIO(no_bg_data))
        else:
            img = Image.open(BytesIO(image_data))
        
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        square_img = self.crop_to_square(img)
        enhanced_img = self.enhance_tire(square_img, image_type=image_type)
        
        output = BytesIO()
        enhanced_img.save(output, format='JPEG', quality=98, subsampling=0)
        output.seek(0)
        
        return output.getvalue()
    
    # ==================== GAUGE METHODS (unchanged from v2) ====================
    # These are imported from v2 to keep gauge processing exactly as-is
    
    def detect_orange_gauge(self, img_array):
        """Detect orange gauge case in image using color detection."""
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h, w = img_array.shape[:2]
        
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        best_contour = None
        best_area = 0
        min_area = (min(w, h) * 0.02) ** 2
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > best_area and area > min_area:
                best_area = area
                best_contour = contour
        
        if best_contour is None:
            return None
        
        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        confidence = min(1.0, best_area / (w * h * 0.01))
        
        logger.info(f"Orange gauge detected at ({cx}, {cy}), conf={confidence:.2f}")
        return (cx, cy, confidence)
    
    def detect_gauge(self, img_array):
        """Detect tread depth gauge in a tire photo."""
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        candidates = []
        search_h = int(h * 0.65)
        upper = gray[:search_h, :]
        
        # Signal 1: LCD Display
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
            
            if contrast > 10 and inside_mean > 40 and border_mean < 130:
                cx_norm = (x + cw // 2) / w
                center_dist = abs(cx_norm - 0.5) * 2
                center_score = 1.0 - center_dist * 0.5
                candidates.append({
                    'type': 'lcd',
                    'cx': x + cw // 2, 'cy': y + ch // 2,
                    'score': contrast * (1.5 if 1.5 < aspect < 4 else 0.5) * center_score
                })
        
        # Signal 2: Blue gauge tip
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
        
        # Signal 3: Mechanical dial gauge
        blurred = cv2.GaussianBlur(gray[:search_h, :], (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                    param1=100, param2=40, minRadius=15, maxRadius=100)
        if circles is not None:
            circles_arr = np.round(circles[0]).astype(int)
            for cx, cy, r in circles_arr:
                mask = np.zeros(gray[:search_h, :].shape, np.uint8)
                cv2.circle(mask, (cx, cy), r, 255, -1)
                mean_bright = np.mean(gray[:search_h, :][mask > 0])
                mean_sat = np.mean(hsv[:search_h, :, 1][mask > 0])
                if mean_bright > 150 and mean_sat < 80:
                    cx_norm = cx / w
                    center_dist = abs(cx_norm - 0.5) * 2
                    center_score = 1.0 - center_dist * 0.5
                    if cx < w * 0.2 or cx > w * 0.8:
                        continue
                    if cy > h * 0.40:
                        continue
                    candidates.append({
                        'type': 'dial', 'cx': cx, 'cy': cy,
                        'radius': r,
                        'score': mean_bright * center_score * 0.5
                    })
        
        if not candidates:
            return None
        
        # Filter by position
        filtered = []
        for c in candidates:
            if c['type'] == 'lcd':
                if c['cx'] < w * 0.15 or c['cx'] > w * 0.85 or c['cy'] > h * 0.50:
                    continue
            if c['type'] == 'blue_tip':
                if c['cx'] < w * 0.2 or c['cx'] > w * 0.8:
                    continue
            filtered.append(c)
        
        if not filtered:
            return None
        candidates = filtered
        
        # Combine signals
        lcds = [c for c in candidates if c['type'] == 'lcd']
        tips = [c for c in candidates if c['type'] == 'blue_tip']
        dials = [c for c in candidates if c['type'] == 'dial']
        
        if lcds:
            if len(lcds) >= 2:
                lcds_sorted = sorted(lcds, key=lambda c: c['cy'])
                y_spread = lcds_sorted[-1]['cy'] - lcds_sorted[0]['cy']
                if y_spread > 120:
                    lcds = []
            
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
                    return (best_pos[0], best_pos[1], confidence, 'digital')
        
        if dials:
            best_dial = max(dials, key=lambda c: c['score'])
            confidence = min(best_dial['score'] / 100, 1.0)
            return (best_dial['cx'], best_dial['cy'], confidence, 'dial', best_dial.get('radius', 40))
        
        return None
    
    def crop_gauge_closeup(self, image_data, gauge_xy=None, output_size=None):
        """
        Crop around the tread depth gauge for a close-up measurement image.
        Kept identical to v2 — gauge processing is separate work.
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
            dial_radius = 40
            
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
                roi_cx, roi_cy = cx - sx1, cy - sy1
                best_dist = float('inf')
                for c in circles[0]:
                    dcx, dcy, dr = int(c[0]), int(c[1]), int(c[2])
                    mask = np.zeros(roi.shape, np.uint8)
                    cv2.circle(mask, (dcx, dcy), dr, 255, -1)
                    mean_bright = np.mean(roi[mask > 0])
                    mean_sat = np.mean(hsv[sy1:sy2, sx1:sx2, 1][mask > 0])
                    if mean_bright > 140 and mean_sat < 90:
                        dist = ((dcx - roi_cx)**2 + (dcy - roi_cy)**2)**0.5
                        if dist < best_dist:
                            best_dist = dist
                            dial_radius = dr
                            cx = sx1 + dcx
                            cy = sy1 + dcy
                            gauge_type = 'dial'
                
                if gauge_type == 'dial':
                    logger.info(f"Manual click snapped to dial: ({cx},{cy}) r={dial_radius}")
        
        # Crop around gauge
        min_dim = min(w, h)
        
        if gauge_type == 'dial':
            target_dial_pct = 0.28
            ideal_side = int(2 * dial_radius / target_dial_pct)
            ideal_side = max(ideal_side, int(min_dim * 0.12))
            ideal_side = min(ideal_side, int(min_dim * 0.45))
            half = ideal_side // 2
        else:
            half = int(min_dim * 0.15)
        
        cy_adjusted = cy + int(half * 0.25) if gauge_type != 'dial' else cy
        
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)
        y1 = max(0, cy_adjusted - half)
        y2 = min(h, cy_adjusted + half)
        
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w != crop_h:
            side = min(crop_w, crop_h)
            x2 = x1 + side
            y2 = y1 + side
        
        crop = np.array(img)[y1:y2, x1:x2]
        crop_img = Image.fromarray(crop)
        
        # Resize to output
        crop_img = crop_img.resize((output_size, output_size), Image.Resampling.LANCZOS)
        
        # Light enhancement
        crop_img = ImageEnhance.Sharpness(crop_img).enhance(1.3)
        crop_img = ImageEnhance.Contrast(crop_img).enhance(1.05)
        
        output = BytesIO()
        crop_img.save(output, format='JPEG', quality=98, subsampling=0)
        output.seek(0)
        
        msg = f"Gauge closeup ({gauge_type}) from ({cx},{cy}), confidence={conf:.2f}"
        logger.info(msg)
        return output.getvalue(), msg
    
    def crop_gauge_auto(self, image_data, output_size=None):
        """Auto-detect orange gauge and crop."""
        output_size = output_size or self.output_size
        
        if isinstance(image_data, bytes):
            img = Image.open(BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, Image.Image):
            img = image_data if image_data.mode == 'RGB' else image_data.convert('RGB')
        else:
            img = image_data
        
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        result = self.detect_orange_gauge(img_array)
        if result is None:
            return None, "No orange gauge case detected"
        
        cx, cy, conf = result
        if conf < 0.3:
            return None, f"Low confidence orange detection ({conf:.2f})"
        
        min_dim = min(w, h)
        crop_radius = int(min_dim * 0.15)
        cy_adjusted = cy + int(crop_radius * 0.3)
        
        x1 = max(0, cx - crop_radius)
        x2 = min(w, cx + crop_radius)
        y1 = max(0, cy_adjusted - crop_radius)
        y2 = min(h, cy_adjusted + crop_radius)
        
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w != crop_h:
            side = min(crop_w, crop_h)
            x2 = x1 + side
            y2 = y1 + side
        
        cropped = img.crop((x1, y1, x2, y2))
        
        final = Image.new('RGB', (output_size, output_size), (255, 255, 255))
        target_size = int(output_size * 0.9)
        cropped_resized = cropped.resize((target_size, target_size), Image.LANCZOS)
        offset = (output_size - target_size) // 2
        final.paste(cropped_resized, (offset, offset))
        
        final = ImageEnhance.Contrast(final).enhance(1.1)
        final = ImageEnhance.Sharpness(final).enhance(1.2)
        
        output = BytesIO()
        final.save(output, format='JPEG', quality=95, subsampling=0)
        output.seek(0)
        
        msg = f"Gauge auto-cropped from ({cx},{cy}), conf={conf:.2f}"
        logger.info(msg)
        return output.getvalue(), msg


# Backwards compatibility
ImageProcessor = ImageProcessorV3
