from __future__ import annotations
import math
import time
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np
import logging
from .utils.data_classes import HistogramInfo, BrightenFactors

class SingleImageProcessor:
    
               
    ISO_FOR_DARK = 1100
    
    def __init__(self, use_renderscript: bool = False):
        self.use_renderscript = use_renderscript
        self.logger = logging.getLogger(__name__)
        self.cached_avg_sample_size = 1
    
    def process_single_image(self, 
                           bitmaps: List[Image.Image],
                           release_bitmaps: bool,
                           output_bitmap: Optional[Image.Image],
                           hdr_alpha: float,
                           n_tiles: int,
                           ce_preserve_blacks: bool,
                           dro_tonemapping_algorithm) -> Image.Image:
        """Process single image for DRO (Dynamic Range Optimization)."""
        self.logger.debug("processSingleImage")
        
        time_s = time.time() * 1000
        
        width, height = bitmaps[0].size
        input_bitmap = bitmaps[0]
        
        if release_bitmaps:
            output_bitmap = input_bitmap
        elif output_bitmap is None:
            output_bitmap = Image.new('RGB', (width, height))
        
                                                         
        if hasattr(dro_tonemapping_algorithm, 'name'):
            algorithm_name = dro_tonemapping_algorithm.name
        else:
            algorithm_name = str(dro_tonemapping_algorithm)
        
        if 'GAINGAMMA' in algorithm_name:
                                                   
            histo = self.compute_histogram(input_bitmap)
            histogram_info = self.get_histogram_info(histo)
            brightness = histogram_info.median_brightness
            max_brightness = histogram_info.max_brightness
            
            self.logger.debug(f"### processSingleImage: time after computeHistogram: {time.time() * 1000 - time_s}")
            self.logger.debug(f"median brightness: {brightness}")
            self.logger.debug(f"max brightness: {max_brightness}")
            
                                      
            brighten_factors = self.compute_brighten_factors(
                False, 0, 0, brightness, max_brightness
            )
            gain = brighten_factors.gain
            gamma = brighten_factors.gamma
            low_x = brighten_factors.low_x
            mid_x = brighten_factors.mid_x
            
            self.logger.debug(f"gain: {gain}")
            self.logger.debug(f"gamma: {gamma}")
            self.logger.debug(f"low_x: {low_x}")
            self.logger.debug(f"mid_x: {mid_x}")
            
                                        
            if (abs(gain - 1.0) > 1e-5 or max_brightness != 255 or abs(gamma - 1.0) > 1e-5):
                self.logger.debug("apply gain/gamma")
                
                output_bitmap = self.apply_dro_brighten(
                    input_bitmap, output_bitmap, gain, gamma, low_x, mid_x, max_brightness
                )
                
                                                                   
                input_bitmap = output_bitmap
                self.logger.debug(f"### processSingleImage: time after dro_brighten: {time.time() * 1000 - time_s}")
        
                                    
        output_bitmap = self.adjust_histogram(
            input_bitmap, output_bitmap, width, height, hdr_alpha, n_tiles, ce_preserve_blacks
        )
        
        self.logger.debug(f"### time for processSingleImage: {time.time() * 1000 - time_s}")
        return output_bitmap
    
    def brighten_image(self, 
                      bitmap: Image.Image, 
                      brightness: int, 
                      max_brightness: int, 
                      brightness_target: int) -> Image.Image:
        """Brighten image to target brightness level."""
        self.logger.debug("brightenImage")
        self.logger.debug(f"brightness: {brightness}")
        self.logger.debug(f"max_brightness: {max_brightness}")
        self.logger.debug(f"brightness_target: {brightness_target}")
        
        brighten_factors = self.compute_brighten_factors(
            False, 0, 0, brightness, max_brightness, brightness_target, False
        )
        gain = brighten_factors.gain
        gamma = brighten_factors.gamma
        low_x = brighten_factors.low_x
        mid_x = brighten_factors.mid_x
        
        self.logger.debug(f"gain: {gain}")
        self.logger.debug(f"gamma: {gamma}")
        self.logger.debug(f"low_x: {low_x}")
        self.logger.debug(f"mid_x: {mid_x}")
        
        if (abs(gain - 1.0) > 1e-5 or max_brightness != 255 or abs(gamma - 1.0) > 1e-5):
            self.logger.debug("apply gain/gamma")
            
            bitmap = self.apply_dro_brighten(
                bitmap, bitmap, gain, gamma, low_x, mid_x, max_brightness
            )
        
        return bitmap
    
    def apply_dro_brighten(self, 
                          input_bitmap: Image.Image, 
                          output_bitmap: Image.Image,
                          gain: float, 
                          gamma: float, 
                          low_x: float, 
                          mid_x: float, 
                          max_brightness: int) -> Image.Image:
        """Apply DRO brightening with gain and gamma correction based on Java implementation."""
                                               
        input_array = np.array(input_bitmap, dtype=np.float32)
        height, width = input_array.shape[:2]
        
                             
        output_array = np.zeros_like(input_array)
        
                                                        
        for y in range(height):
            for x in range(width):
                pixel = input_array[y, x]
                
                                                        
                if len(pixel.shape) == 0 or len(pixel) == 1:
                               
                    channels = [pixel] if len(pixel.shape) == 0 else pixel
                else:
                           
                    channels = pixel[:3] if len(pixel) >= 3 else pixel
                
                output_pixel = np.zeros_like(pixel)
                
                for c in range(len(channels)):
                    in_value = channels[c] / 255.0                      
                    
                                
                    gained_value = in_value * gain
                    
                                                                                  
                    if gained_value <= low_x / 255.0:
                                            
                        out_value = gained_value
                    elif gained_value <= mid_x / 255.0:
                                                                        
                        low_norm = low_x / 255.0
                        mid_norm = mid_x / 255.0
                        t = (gained_value - low_norm) / (mid_norm - low_norm)
                        
                                                              
                        if gamma != 1.0:
                            gamma_corrected = np.power(t, 1.0 / gamma)
                            out_value = low_norm + gamma_corrected * (mid_norm - low_norm)
                        else:
                            out_value = gained_value
                    else:
                                                                          
                        max_norm = max_brightness / 255.0
                        if gained_value <= max_norm and max_norm > 0:
                                                           
                            scaled_value = gained_value / max_norm
                            if gamma != 1.0:
                                gamma_corrected = np.power(scaled_value, 1.0 / gamma)
                                out_value = gamma_corrected * max_norm
                            else:
                                out_value = gained_value
                        else:
                                             
                            out_value = max_norm if max_norm > 0 else 1.0
                    
                                                             
                    output_pixel[c] = np.clip(out_value * 255.0, 0, 255)
                
                output_array[y, x] = output_pixel
        
                               
        output_array = output_array.astype(np.uint8)
        return Image.fromarray(output_array)
    
    def compute_histogram(self, bitmap: Image.Image, histogram_type: str = "VALUE") -> np.ndarray:
        """Compute histogram of image."""
        if histogram_type == "VALUE":
                                                        
            gray_image = bitmap.convert('L')
            gray_array = np.array(gray_image)
            histogram, _ = np.histogram(gray_array, bins=256, range=(0, 256))
            return histogram
        elif histogram_type == "INTENSITY":
                                
            rgb_array = np.array(bitmap)
            if len(rgb_array.shape) == 3:
                intensity = np.mean(rgb_array, axis=2)
            else:
                intensity = rgb_array
            histogram, _ = np.histogram(intensity, bins=256, range=(0, 256))
            return histogram
        elif histogram_type == "LUMINANCE":
                                                              
            rgb_array = np.array(bitmap)
            if len(rgb_array.shape) == 3:
                luminance = (0.299 * rgb_array[:,:,0] + 
                           0.587 * rgb_array[:,:,1] + 
                           0.114 * rgb_array[:,:,2])
            else:
                luminance = rgb_array
            histogram, _ = np.histogram(luminance, bins=256, range=(0, 256))
            return histogram
        else:
                              
            return self.compute_histogram(bitmap, "VALUE")
    
    def get_histogram_info(self, histogram: np.ndarray) -> HistogramInfo:
        """Extract histogram information."""
        total = np.sum(histogram)
        
                                        
        cumsum = np.cumsum(histogram)
        
                                
        median_idx = np.where(cumsum >= total // 2)[0][0] if total > 0 else 127
        median_brightness = int(median_idx)
        
                                                 
        max_brightness = 255
        for i in range(255, -1, -1):
            if histogram[i] > 0:
                max_brightness = i
                break
        
                                   
        sum_brightness = sum(histogram[i] * i for i in range(256))
        mean_brightness = int(sum_brightness / total + 0.1) if total > 0 else 127
        
        return HistogramInfo(total, mean_brightness, median_brightness, max_brightness)
    
    def compute_brighten_factors(self, 
                                is_hdr: bool,
                                hdr_median: int,
                                hdr_max: int,
                                brightness: int,
                                max_brightness: int,
                                brightness_target: Optional[int] = None,
                                brighten_only: bool = True) -> BrightenFactors:
        """Compute gain and gamma factors for brightening based on Java implementation."""
        
                                                     
        if brightness_target is None:
            max_gain_factor = 1.5
            ideal_brightness = 119
            brightness_target = self.get_brightness_target(brightness, max_gain_factor, ideal_brightness)
        
        if brightness <= 0:
            brightness = 1
            
                                
        gain = brightness_target / brightness
        
        if gain < 1.0 and brighten_only:
            gain = 1.0
        
                          
        gamma = 1.0
        max_possible_value = gain * max_brightness
        
                                                           
        mid_x = 255.5
        if max_possible_value > 255.0:
            self.logger.debug("use piecewise gain/gamma")
                                                        
                                                                  
            mid_y_factor = 0.8                                            
            mid_y = mid_y_factor * 255.0
            mid_x = mid_y / gain
            if max_brightness > 0:
                gamma = math.log(mid_y / 255.0) / math.log(mid_x / max_brightness)
        elif brighten_only and max_possible_value < 255.0 and max_brightness > 0:
                                                            
            alt_gain = 255.0 / max_brightness
            alt_gain = min(alt_gain, 4.0)                  
            if alt_gain > gain:
                gain = alt_gain
                self.logger.debug(f"increased gain to: {gain}")
        
                                               
        low_x = 0.0
        if is_hdr or max_brightness < 200:                        
                                                     
            piecewise_mid_y = 0.5 * 255.0
            piecewise_mid_x = piecewise_mid_y / gain
            low_x = min(8.0, 0.125 * piecewise_mid_x)
        
        self.logger.debug(f"gain: {gain}")
        self.logger.debug(f"gamma: {gamma}")
        self.logger.debug(f"low_x: {low_x}")
        self.logger.debug(f"mid_x: {mid_x}")
        
        return BrightenFactors(gain, gamma, low_x, mid_x)
    
    def get_brightness_target(self, brightness: int, max_gain_factor: float, ideal_brightness: int) -> int:
        """Calculate target brightness with gain limiting."""
        if brightness > 0:
                                       
            min_brightness_c = 42.0
            min_max_gain_factor = min_brightness_c / brightness
            max_gain_factor = max(max_gain_factor, min_max_gain_factor)
            max_gain_factor = min(max_gain_factor, 15.0)                    
        
        if brightness <= 0:
            brightness = 1
            
        median_target = min(ideal_brightness, int(max_gain_factor * brightness))
        return max(brightness, median_target)                     
    
    def adjust_histogram(self, 
                        input_bitmap: Image.Image,
                        output_bitmap: Image.Image,
                        width: int,
                        height: int,
                        hdr_alpha: float,
                        n_tiles: int,
                        ce_preserve_blacks: bool) -> Image.Image:
        """Apply histogram adjustment for contrast enhancement."""
        if n_tiles <= 1:
            return self.global_histogram_adjustment(input_bitmap, hdr_alpha, ce_preserve_blacks)
        else:
            return self.local_histogram_adjustment(input_bitmap, hdr_alpha, n_tiles, ce_preserve_blacks)
    
    def global_histogram_adjustment(self, 
                                  image: Image.Image, 
                                  alpha: float, 
                                  preserve_blacks: bool) -> Image.Image:
        """Apply global histogram adjustment."""
        input_array = np.array(image, dtype=np.float32)
        
                                        
        mean_val = np.mean(input_array)
        std_val = np.std(input_array)
        
                                    
        contrast_factor = 1.0 + alpha * 0.8
        enhanced = (input_array - mean_val) * contrast_factor + mean_val
        
        if preserve_blacks:
                                     
            threshold = mean_val * 0.4
            mask = input_array < threshold
            enhanced[mask] = input_array[mask] * (1.0 + alpha * 0.2)
        
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        return Image.fromarray(enhanced)
    
    def local_histogram_adjustment(self, 
                                 image: Image.Image, 
                                 alpha: float, 
                                 n_tiles: int, 
                                 preserve_blacks: bool) -> Image.Image:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) based on Java implementation."""
        
        image_array = np.array(image)
        width, height = image.size
        
                                                    
        c_histogram = np.zeros(n_tiles * n_tiles * 256, dtype=np.int32)
        temp_histogram = np.zeros(256, dtype=np.int32)
        
                           
        for i in range(n_tiles):
            a0 = i / n_tiles
            a1 = (i + 1.0) / n_tiles
            start_x = int(a0 * width)
            stop_x = int(a1 * width)
            
            if stop_x == start_x:
                continue
                
            for j in range(n_tiles):
                b0 = j / n_tiles
                b1 = (j + 1.0) / n_tiles
                start_y = int(b0 * height)
                stop_y = int(b1 * height)
                
                if stop_y == start_y:
                    continue
                
                              
                tile = image_array[start_y:stop_y, start_x:stop_x]
                
                                                 
                histogram = self.compute_tile_histogram(tile)
                
                                                      
                self.clip_histogram(histogram, temp_histogram, 
                                  stop_x - start_x, stop_y - start_y, preserve_blacks)
                
                                              
                histogram_offset = 256 * (i * n_tiles + j)
                c_histogram[histogram_offset] = histogram[0]
                for k in range(1, 256):
                    c_histogram[histogram_offset + k] = c_histogram[histogram_offset + k - 1] + histogram[k]
        
                                             
        output_array = self.apply_histogram_adjustment(
            image_array, c_histogram, n_tiles, width, height, alpha
        )
        
        return Image.fromarray(output_array)
    
    def compute_tile_histogram(self, tile: np.ndarray) -> np.ndarray:
        """Compute histogram for a tile based on max RGB value."""
        histogram = np.zeros(256, dtype=np.int32)
        
        if len(tile.shape) == 3:
                                                                                       
            for y in range(tile.shape[0]):
                for x in range(tile.shape[1]):
                    pixel = tile[y, x]
                    if len(pixel) >= 3:
                                                              
                        value = int(max(pixel[0], pixel[1], pixel[2]))
                    else:
                        value = int(pixel[0]) if len(pixel) > 0 else 0
                    value = max(0, min(255, value))
                    histogram[value] += 1
        else:
                             
            flat_tile = tile.flatten()
            for value in flat_tile:
                val = max(0, min(255, int(value)))
                histogram[val] += 1
        
        return histogram
    
    def clip_histogram(self, histogram: np.ndarray, temp_histogram: np.ndarray,
                      sub_width: int, sub_height: int, preserve_blacks: bool):
        """Clip histogram for CLAHE algorithm - based on Java implementation."""
        n_pixels = sub_width * sub_height
        clip_limit = (5 * n_pixels) // 256
        
                                                  
        bottom = 0
        top = clip_limit
        while top - bottom > 1:
            middle = (top + bottom) // 2
            total_excess = sum(max(0, histogram[x] - middle) for x in range(256))
            if total_excess > (clip_limit - middle) * 256:
                top = middle
            else:
                bottom = middle
        
        clip_limit = (top + bottom) // 2
        
                                                 
        n_clipped = 0
        for x in range(256):
            if histogram[x] > clip_limit:
                n_clipped += histogram[x] - clip_limit
                histogram[x] = clip_limit
        
                                            
        n_clipped_per_bucket = n_clipped // 256
        for x in range(256):
            histogram[x] += n_clipped_per_bucket
        
        if preserve_blacks:
                                                          
            temp_histogram[0] = histogram[0]
            for x in range(1, 256):
                temp_histogram[x] = temp_histogram[x-1] + histogram[x]
            
            equal_limit = n_pixels // 256
            dark_threshold = 128
            
            for x in range(dark_threshold):
                c_equal_limit = equal_limit * (x + 1)
                if temp_histogram[x] >= c_equal_limit:
                    continue
                
                alpha = 1.0 - (x / dark_threshold)
                limit = int(alpha * equal_limit)
                
                if histogram[x] < limit:
                                                       
                    for y in range(x + 1, 256):
                        if histogram[x] >= limit:
                            break
                        if histogram[y] > equal_limit:
                            move = min(histogram[y] - equal_limit, limit - histogram[x])
                            histogram[x] += move
                            histogram[y] -= move
    
    def apply_histogram_adjustment(self, image_array: np.ndarray, c_histogram: np.ndarray,
                                 n_tiles: int, width: int, height: int, alpha: float) -> np.ndarray:
        """Apply histogram adjustment to image using cumulative histograms."""
        output_array = np.zeros_like(image_array, dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                                            
                tile_x = min(int((x / width) * n_tiles), n_tiles - 1)
                tile_y = min(int((y / height) * n_tiles), n_tiles - 1)
                
                pixel = image_array[y, x]
                
                                                        
                if len(pixel.shape) == 0:
                                  
                    channels = [pixel]
                elif len(pixel) == 1:
                               
                    channels = [pixel[0]]
                else:
                                                  
                    channels = pixel[:3] if len(pixel) >= 3 else pixel
                
                output_pixel = np.zeros_like(pixel, dtype=np.float32)
                
                for c in range(len(channels)):
                    value = int(np.clip(channels[c], 0, 255))
                    
                                                            
                    histogram_offset = 256 * (tile_y * n_tiles + tile_x)
                    cum_value = c_histogram[histogram_offset + value]
                    
                                         
                    tile_start_x = int((tile_x / n_tiles) * width)
                    tile_stop_x = int(((tile_x + 1) / n_tiles) * width)
                    tile_start_y = int((tile_y / n_tiles) * height)
                    tile_stop_y = int(((tile_y + 1) / n_tiles) * height)
                    
                    tile_size = (tile_stop_x - tile_start_x) * (tile_stop_y - tile_start_y)
                    
                                                  
                    if tile_size > 0:
                        equalized = (cum_value * 255) / tile_size
                                                             
                        result = (1.0 - alpha) * value + alpha * equalized
                        output_pixel[c] = np.clip(result, 0, 255)
                    else:
                        output_pixel[c] = value
                
                if len(pixel.shape) == 0:
                    output_array[y, x] = output_pixel[0]
                else:
                    output_array[y, x] = output_pixel
        
        return np.clip(output_array, 0, 255).astype(np.uint8)
    
    @staticmethod
    def scene_is_low_light(iso: int, exposure_time: int) -> bool:
        """Determine if scene is low light based on ISO and exposure time."""
        ISO_FOR_DARK = SingleImageProcessor.ISO_FOR_DARK
        
                                                                             
        iso_exposure_condition = (iso >= ISO_FOR_DARK and 
                                iso * exposure_time >= 69 * 1000000000)
        long_exposure_condition = exposure_time >= (1000000000 // 5 - 10000)
        
        return iso_exposure_condition or long_exposure_condition
    
    def get_avg_sample_size(self, capture_result_iso: int, capture_result_exposure_time: int) -> int:
        """Get sample size for averaging based on scene conditions."""
        self.cached_avg_sample_size = 2 if self.scene_is_low_light(
            capture_result_iso, capture_result_exposure_time
        ) else 1
        
        self.logger.debug(f"getAvgSampleSize: {self.cached_avg_sample_size}")
        return self.cached_avg_sample_size
    
    def get_avg_sample_size_cached(self) -> int:
        """Get cached average sample size."""
        return self.cached_avg_sample_size
    
    def average_images(self, 
                      bitmap_avg: Image.Image, 
                      bitmap_new: Image.Image, 
                      avg_factor: float,
                      iso: int,
                      exposure_time: int,
                      zoom_factor: float = 1.0) -> Image.Image:
        """
        Combines two images by averaging them. Each pixel of bitmap_avg is modified to contain:
        (avg_factor * bitmap_avg + bitmap_new)/(avg_factor+1)
        
        Args:
            bitmap_avg: One of the input images (will be modified)
            bitmap_new: The other input image
            avg_factor: The weighting factor for bitmap_avg
            iso: The ISO used to take the photos
            exposure_time: The exposure time used to take the photos  
            zoom_factor: The digital zoom factor used to take the photos
        """
                                        
        avg_array = np.array(bitmap_avg, dtype=np.float32)
        new_array = np.array(bitmap_new, dtype=np.float32)
        
                                        
        result_array = (avg_factor * avg_array + new_array) / (avg_factor + 1.0)
        
                               
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        return Image.fromarray(result_array)
    
    def compute_black_level(self, histogram_info: HistogramInfo, histo: np.ndarray, iso: int) -> float:
        """Compute black level for dehaze algorithm based on Java implementation."""
        black_level = 0.0
        
                                          
        total = histogram_info.total
        percentile = int(total * 0.001)                   
        count = 0
        darkest_brightness = -1
        
        for i in range(len(histo)):
            count += histo[i]
            if count >= percentile and darkest_brightness == -1:
                darkest_brightness = i
                break
        
        if darkest_brightness != -1:
            black_level = max(black_level, darkest_brightness)
        
                                                                     
        max_black_level = 18 if iso <= 700 else 4
        black_level = min(black_level, max_black_level)
        
        self.logger.debug(f"black_level: {black_level}")
        return black_level
    
    def apply_median_filter(self, image_array: np.ndarray, strength: float) -> np.ndarray:
        """Apply median filter for noise reduction."""
        if strength <= 0:
            return image_array
        
        height, width = image_array.shape[:2]
        result = image_array.copy()
        
                                  
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if len(image_array.shape) == 3:
                                 
                    for c in range(image_array.shape[2]):
                        window = image_array[y-1:y+2, x-1:x+2, c].flatten()
                        median_val = np.median(window)
                                                               
                        result[y, x, c] = (1.0 - strength) * image_array[y, x, c] + strength * median_val
                else:
                               
                    window = image_array[y-1:y+2, x-1:x+2].flatten()
                    median_val = np.median(window)
                    result[y, x] = (1.0 - strength) * image_array[y, x] + strength * median_val
        
        return result
    
    def final_brighten_processing(self, 
                                 image: Image.Image,
                                 iso: int,
                                 exposure_time: int,
                                 apply_contrast_enhancement: bool = True) -> Image.Image:
        """Final processing stage for noise reduction and contrast enhancement."""
        if not apply_contrast_enhancement:
            return image
            
                                                                                     
        bright_scene_threshold_iso = 1100
        bright_scene_threshold_exposure = 1000000000 // 59               
        
        is_bright_scene = iso < bright_scene_threshold_iso and exposure_time < bright_scene_threshold_exposure
        
        if is_bright_scene:
                                                               
            histo = self.compute_histogram(image)
            histogram_info = self.get_histogram_info(histo)
            
                                                                     
            median_lo = 60
            median_hi = 35
            alpha = (histogram_info.median_brightness - median_lo) / (median_hi - median_lo)
            alpha = max(0.0, min(1.0, alpha))
            amount = (1.0 - alpha) * 0.25 + alpha * 0.5
            
            self.logger.debug(f"contrast enhancement alpha: {alpha}")
            self.logger.debug(f"contrast enhancement amount: {amount}")
            
                                        
            image = self.adjust_histogram(image, image, image.width, image.height, amount, 1, True)
        
        return image