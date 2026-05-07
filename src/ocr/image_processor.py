"""
Image Processor Module
Handles preprocessing of medical lab result images for OCR.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from typing import Tuple, Optional, Union
import io


class ImageProcessor:
    """
    Process medical lab images to improve OCR accuracy.
    """
    
    def __init__(self):
        self.target_dpi = 300
        self.min_size = (800, 600)
        
    def load_image(self, image_path: Union[str, Path, bytes]) -> np.ndarray:
        """
        Load image from file path or bytes.
        
        Args:
            image_path: Path to image file or image bytes
            
        Returns:
            OpenCV image (BGR format)
        """
        if isinstance(image_path, bytes):
            # Load from bytes
            nparr = np.frombuffer(image_path, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Load from file
            image = cv2.imread(str(image_path))
            
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        return image
    
    def resize_image(
        self,
        image: np.ndarray,
        min_width: int = 800,
        max_width: int = 2000
    ) -> np.ndarray:
        """Resize image to optimal size for OCR."""
        height, width = image.shape[:2]
        
        if width < min_width:
            # Scale up
            scale = min_width / width
        elif width > max_width:
            # Scale down
            scale = max_width / width
        else:
            return image
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    def rotate_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct orientation of rotated images."""
        # Simple check: if height > width significantly, might be rotated
        height, width = image.shape[:2]
        
        # For lab reports, typically width > height (landscape or close)
        # If portrait with extreme ratio, rotate
        if height > width * 1.5:
            # Rotate 90 degrees clockwise
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
        return image
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image using bilateral filter."""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def apply_thresholding(
        self,
        image: np.ndarray,
        method: str = 'adaptive'
    ) -> np.ndarray:
        """
        Apply thresholding to enhance text.
        
        Args:
            image: Grayscale image
            method: 'adaptive', 'otsu', or 'simple'
        """
        if method == 'adaptive':
            return cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
        elif method == 'otsu':
            _, thresh = cv2.threshold(
                image, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return thresh
        else:
            _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return thresh
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct skew in the image."""
        # Find all non-zero points (text)
        coords = np.column_stack(np.where(image > 0))
        
        if len(coords) < 100:
            return image
            
        # Calculate the angle
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
            
        # Rotate image to deskew
        if abs(angle) > 0.5:  # Only rotate if skew is significant
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
        return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image to make text clearer."""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Remove shadows from image (useful for photos)."""
        if len(image.shape) == 3:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Morphological operations to estimate background
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        
        # Divide to remove shadow
        result = cv2.divide(gray, bg, scale=255)
        
        return result
    
    def preprocess_for_ocr(
        self,
        image: Union[np.ndarray, str, Path, bytes],
        remove_shadows: bool = True,
        apply_threshold: bool = False
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline for OCR.
        
        Args:
            image: Input image (array, path, or bytes)
            remove_shadows: Apply shadow removal (for photos)
            apply_threshold: Apply binary thresholding
            
        Returns:
            Preprocessed image ready for OCR
        """
        # Load if necessary
        if isinstance(image, (str, Path, bytes)):
            image = self.load_image(image)
        
        # Resize to optimal size
        image = self.resize_image(image)
        
        # Rotate if needed
        image = self.rotate_if_needed(image)
        
        # Enhance contrast
        image = self.enhance_contrast(image)
        
        # Convert to grayscale
        gray = self.convert_to_grayscale(image)
        
        # Remove shadows if requested
        if remove_shadows:
            gray = self.remove_shadows(gray)
        
        # Remove noise
        gray = self.remove_noise(gray)
        
        # Apply thresholding if requested
        if apply_threshold:
            gray = self.apply_thresholding(gray, method='adaptive')
        
        return gray
    
    def get_image_for_display(
        self,
        image: np.ndarray,
        max_size: int = 800
    ) -> np.ndarray:
        """Resize image for display purposes."""
        height, width = image.shape[:2]
        
        if max(height, width) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            image = cv2.resize(image, (new_width, new_height))
        
        return image
    
    def save_processed_image(
        self,
        image: np.ndarray,
        output_path: Union[str, Path]
    ):
        """Save processed image to file."""
        cv2.imwrite(str(output_path), image)
        print(f"Saved processed image to {output_path}")


if __name__ == "__main__":
    # Test image processor
    processor = ImageProcessor()
    
    # Test with a sample image path
    sample_path = "sample_images/test_lab_result.jpg"
    
    if Path(sample_path).exists():
        processed = processor.preprocess_for_ocr(sample_path)
        print(f"Processed image shape: {processed.shape}")
    else:
        print("No sample image found for testing")
