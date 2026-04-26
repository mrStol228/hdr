from __future__ import annotations
import math
import time
from typing import List, Optional, Tuple, Callable
from PIL import Image
import numpy as np
import logging
from .utils.data_classes import BrightnessDetails, LuminanceInfo
from .utils.exceptions import HDRProcessorException

class MultiImageProcessor:
    """
    Enhanced multi-image processor with advanced MTB alignment and robust algorithms.
    Улучшенный процессор многих изображений с продвинутым MTB выравниванием и робастными алгоритмами.
    """
    
    def __init__(self, use_renderscript: bool = False):
        self.use_renderscript = use_renderscript
        self.logger = logging.getLogger(__name__)
        self.offsets_x = None
        self.offsets_y = None
    
    def process_avg_multi(self, 
                         bitmaps: List[Image.Image],
                         hdr_alpha: float,
                         n_tiles: int,
                         ce_preserve_blacks: bool):
        """
        Combines multiple images by averaging them with advanced alignment.
        
        Args:
            bitmaps: Input bitmaps. The resultant bitmap will be stored as the first bitmap on exit
            hdr_alpha: HDR effect strength
            n_tiles: Number of tiles for local histogram adjustment
            ce_preserve_blacks: Whether to preserve blacks in contrast enhancement
        """
        self.logger.debug("processAvgMulti")
        self.logger.debug(f"hdr_alpha: {hdr_alpha}")
        
        n_bitmaps = len(bitmaps)
        if n_bitmaps != 8:
            self.logger.error(f"n_bitmaps should be 8, not {n_bitmaps}")
            raise HDRProcessorException(HDRProcessorException.INVALID_N_IMAGES)
        
                                               
        base_size = bitmaps[0].size
        for i in range(1, n_bitmaps):
            if bitmaps[i].size != base_size:
                self.logger.error("bitmaps not of same resolution")
                for j in range(n_bitmaps):
                    self.logger.error(f"bitmaps {j}: {bitmaps[j].size}")
                raise HDRProcessorException(HDRProcessorException.UNEQUAL_SIZES)
        
        time_s = time.time() * 1000
        width, height = base_size
        
        self.logger.debug(f"### time after validation: {time.time() * 1000 - time_s}")
        
                                              
        offsets_x, offsets_y = self.auto_alignment_multi_image(
            bitmaps, base_bitmap=0, use_mtb=True, max_align_scale=2, time_s=time_s
        )
        
        self.logger.debug(f"### time after alignment: {time.time() * 1000 - time_s}")
        
                                            
        image_arrays = []
        for i, bitmap in enumerate(bitmaps):
            array = np.array(bitmap, dtype=np.float32)
            image_arrays.append(array)
        
        self.logger.debug(f"### time after creating arrays: {time.time() * 1000 - time_s}")
        
                                                                
        result_array = self.avg_multi_function_aligned(image_arrays, offsets_x, offsets_y, width, height)
        
        self.logger.debug(f"### time after averaging: {time.time() * 1000 - time_s}")
        
                                              
        if hdr_alpha != 0.0:
            result_array = self.adjust_histogram_array(
                result_array, width, height, hdr_alpha, n_tiles, ce_preserve_blacks
            )
            self.logger.debug(f"### time after adjustHistogram: {time.time() * 1000 - time_s}")
        
                                                           
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        bitmaps[0] = Image.fromarray(result_array)
        
                                
        for i in range(1, len(bitmaps)):
            bitmaps[i] = None
        
        self.logger.debug(f"### time for processAvgMulti: {time.time() * 1000 - time_s}")
    
    def auto_alignment_multi_image(self, bitmaps: List[Image.Image], base_bitmap: int, 
                                 use_mtb: bool, max_align_scale: int, time_s: float) -> Tuple[List[int], List[int]]:
        """
        Perform auto-alignment on multiple images using advanced MTB algorithm.
        """
        self.logger.debug("auto_alignment_multi_image")
        
        n_bitmaps = len(bitmaps)
        offsets_x = [0] * n_bitmaps
        offsets_y = [0] * n_bitmaps
        
                    
        width, height = bitmaps[0].size
        
                                                                         
        mtb_width = width // 2
        mtb_height = height // 2
        mtb_x = mtb_width // 2
        mtb_y = mtb_height // 2
        
        self.logger.debug(f"mtb region: {mtb_x}, {mtb_y}, {mtb_width}, {mtb_height}")
        
                                                            
        luminance_infos = []
        if use_mtb:
            for i in range(n_bitmaps):
                luminance_info = self.compute_median_luminance(
                    bitmaps[i], mtb_x, mtb_y, mtb_width, mtb_height
                )
                luminance_infos.append(luminance_info)
                self.logger.debug(f"Image {i} median_value: {luminance_info.median_value}")
        
        self.logger.debug(f"time after computeMedianLuminance: {time.time() * 1000 - time_s}")
        
                                                             
        mtb_images = []
        for i in range(n_bitmaps):
            if use_mtb and not luminance_infos[i].noisy:
                median_value = luminance_infos[i].median_value
                                                          
                min_diff_c = 4
                median_value = max(median_value, min_diff_c + 1)
                median_value = min(median_value, 255 - (min_diff_c + 1))
                
                mtb_image = self.create_mtb_image(bitmaps[i], median_value, mtb_x, mtb_y, mtb_width, mtb_height)
            else:
                                       
                mtb_image = self.create_grayscale_image(bitmaps[i], mtb_x, mtb_y, mtb_width, mtb_height)
            
            mtb_images.append(mtb_image)
        
        self.logger.debug(f"### time after creating MTB images: {time.time() * 1000 - time_s}")
        
                                                       
        if mtb_images[base_bitmap] is None:
            self.logger.debug("base image not suitable for alignment")
            median_brightness = luminance_infos[base_bitmap].median_value if use_mtb else 127
            return offsets_x, offsets_y
        
                                       
        max_dim = max(width, height)
        max_ideal_size = (max_align_scale * max_dim) // 150
        initial_step_size = 1
        while initial_step_size < max_ideal_size:
            initial_step_size *= 2
        
        self.logger.debug(f"max_ideal_size: {max_ideal_size}")
        self.logger.debug(f"initial_step_size: {initial_step_size}")
        
                                        
        for i in range(n_bitmaps):
            if i == base_bitmap:
                continue
            
            if mtb_images[i] is None:
                self.logger.debug(f"image {i} not suitable for alignment")
                continue
            
                                            
            offset_x, offset_y = self.hierarchical_mtb_alignment(
                mtb_images[base_bitmap], mtb_images[i], 
                initial_step_size, use_mtb, mtb_width, mtb_height
            )
            
            offsets_x[i] = offset_x
            offsets_y[i] = offset_y
            
            self.logger.debug(f"Image {i} final offsets: x={offset_x}, y={offset_y}")
        
        return offsets_x, offsets_y
    
    def compute_median_luminance(self, bitmap: Image.Image, mtb_x: int, mtb_y: int, 
                               mtb_width: int, mtb_height: int) -> LuminanceInfo:
        """
        Compute median luminance of image region with noise detection.
        """
        self.logger.debug(f"computeMedianLuminance for region {mtb_x}, {mtb_y}, {mtb_width}, {mtb_height}")
        
        n_samples_c = 100
        n_w_samples = int(math.sqrt(n_samples_c))
        n_h_samples = n_samples_c // n_w_samples
        
        histo = np.zeros(256, dtype=np.int32)
        total = 0
        
                                         
        for y in range(n_h_samples):
            alpha = (y + 1.0) / (n_h_samples + 1.0)
            y_coord = mtb_y + int(alpha * mtb_height)
            
            for x in range(n_w_samples):
                beta = (x + 1.0) / (n_w_samples + 1.0)
                x_coord = mtb_x + int(beta * mtb_width)
                
                                                              
                pixel = bitmap.getpixel((x_coord, y_coord))
                if isinstance(pixel, int):
                    luminance = pixel
                else:
                    r, g, b = pixel[:3]
                    luminance = max(r, g, b)
                
                histo[luminance] += 1
                total += 1
        
                          
        min_value = -1
        hi_value = -1
        noisy = False
        
                                                      
        count_hi = 0
        for i in range(255, -1, -1):
            count_hi += histo[i]
            if count_hi >= total // 10:
                hi_value = i
                break
        
                                   
        middle = total // 2
        count = 0
        for i in range(256):
            count += histo[i]
            
            if min_value == -1 and histo[i] > 0:
                min_value = i
            
            if count >= middle:
                median_value = i
                
                                                             
                noise_threshold = 4
                n_below = sum(histo[j] for j in range(max(0, i - noise_threshold + 1)))
                frac_below = n_below / total
                
                if frac_below < 0.2:
                    self.logger.debug("region too dark/noisy")
                    noisy = True
                
                return LuminanceInfo(min_value, median_value, hi_value, noisy)
        
        self.logger.error("computeMedianLuminance failed")
        return LuminanceInfo(min_value if min_value != -1 else 0, 127, 
                           hi_value if hi_value != -1 else 255, True)
    
    def create_mtb_image(self, bitmap: Image.Image, median_value: int, 
                        mtb_x: int, mtb_y: int, mtb_width: int, mtb_height: int) -> Optional[np.ndarray]:
        """
        Create Median Threshold Bitmap for alignment.
        """
                                                 
        region = bitmap.crop((mtb_x, mtb_y, mtb_x + mtb_width, mtb_y + mtb_height))
        gray_array = np.array(region.convert('L'))
        
                    
        mtb_array = np.zeros_like(gray_array, dtype=np.uint8)
        
                                                  
        min_diff_c = 4                                      
        
        for y in range(gray_array.shape[0]):
            for x in range(gray_array.shape[1]):
                value = gray_array[y, x]
                diff = abs(value - median_value)
                
                if diff <= min_diff_c:
                                                                
                    mtb_array[y, x] = 127                               
                elif value <= median_value:
                    mtb_array[y, x] = 0         
                else:
                    mtb_array[y, x] = 255         
        
        return mtb_array
    
    def create_grayscale_image(self, bitmap: Image.Image, mtb_x: int, mtb_y: int, 
                             mtb_width: int, mtb_height: int) -> np.ndarray:
        """
        Create grayscale image for alignment fallback.
        """
        region = bitmap.crop((mtb_x, mtb_y, mtb_x + mtb_width, mtb_y + mtb_height))
        return np.array(region.convert('L'))
    
    def hierarchical_mtb_alignment(self, mtb_base: np.ndarray, mtb_target: np.ndarray,
                                 initial_step_size: int, use_mtb: bool, 
                                 mtb_width: int, mtb_height: int) -> Tuple[int, int]:
        """
        Perform hierarchical MTB alignment with pyramid approach.
        """
        best_offset_x = 0
        best_offset_y = 0
        
                                                        
        step_size = initial_step_size
        min_step_size = 1
        
        while step_size >= min_step_size:
            pixel_step_size = step_size
            if pixel_step_size > mtb_width or pixel_step_size > mtb_height:
                pixel_step_size = step_size
            
            self.logger.debug(f"Alignment step_size: {step_size}, pixel_step_size: {pixel_step_size}")
            
                                     
            stop_x = mtb_width // pixel_step_size
            stop_y = mtb_height // pixel_step_size
            
                                                      
            errors = self.calculate_alignment_errors(
                mtb_base, mtb_target, best_offset_x, best_offset_y, 
                pixel_step_size, stop_x, stop_y, use_mtb
            )
            
                              
            best_error = float('inf')
            best_id = -1
            
            for j in range(9):
                error = errors[j]
                self.logger.debug(f"    errors[{j}]: {error}")
                
                if error < best_error:
                    best_error = error
                    best_id = j
            
            self.logger.debug(f"    best_id {best_id} error: {best_error}")
            
                                                        
            if best_error >= 2000000000:
                self.logger.error("auto-alignment failed due to overflow")
                best_id = 4                     
            
                                                              
            if best_id != -1:
                this_off_x = (best_id % 3) - 1                       
                this_off_y = (best_id // 3) - 1
                
                best_offset_x += this_off_x * step_size
                best_offset_y += this_off_y * step_size
                
                self.logger.debug(f"    offset update: dx={this_off_x * step_size}, dy={this_off_y * step_size}")
                self.logger.debug(f"    new total offset: x={best_offset_x}, y={best_offset_y}")
            
            step_size //= 2
        
        return best_offset_x, best_offset_y
    
    def calculate_alignment_errors(self, mtb_base: np.ndarray, mtb_target: np.ndarray,
                                 base_offset_x: int, base_offset_y: int, pixel_step_size: int,
                                 stop_x: int, stop_y: int, use_mtb: bool) -> List[int]:
        """
        Calculate alignment errors for 3x3 grid of offsets around current best position.
        """
        errors = [0] * 9
        mtb_height, mtb_width = mtb_base.shape
        
                                   
        offset_deltas = [
            (-pixel_step_size, -pixel_step_size), (0, -pixel_step_size), (pixel_step_size, -pixel_step_size),
            (-pixel_step_size, 0), (0, 0), (pixel_step_size, 0),
            (-pixel_step_size, pixel_step_size), (0, pixel_step_size), (pixel_step_size, pixel_step_size)
        ]
        
        if use_mtb:
                                                     
            for cy in range(stop_y):
                y = cy * pixel_step_size
                y_base = y + base_offset_y
                
                                                
                if y_base < 0 or y_base >= mtb_height:
                    continue
                
                for cx in range(stop_x):
                    x = cx * pixel_step_size
                    x_base = x + base_offset_x
                    
                                                    
                    if x_base < 0 or x_base >= mtb_width:
                        continue
                    
                                             
                    pixel_base = mtb_base[y_base, x_base]
                    
                                                 
                    for i, (dx, dy) in enumerate(offset_deltas):
                        target_x = x_base + dx
                        target_y = y_base + dy
                        
                                                          
                        if (target_x >= 0 and target_x < mtb_width and 
                            target_y >= 0 and target_y < mtb_height):
                            
                            pixel_target = mtb_target[target_y, target_x]
                            
                                                                      
                            if (pixel_base != pixel_target and 
                                pixel_base != 127 and pixel_target != 127):
                                errors[i] += 1
        else:
                                                  
            overflow_check_c = 2000000000
            
            for cy in range(stop_y):
                y = cy * pixel_step_size
                y_base = y + base_offset_y
                
                if y_base < 0 or y_base >= mtb_height:
                    continue
                
                for cx in range(stop_x):
                    x = cx * pixel_step_size
                    x_base = x + base_offset_x
                    
                    if x_base < 0 or x_base >= mtb_width:
                        continue
                    
                    pixel_base = mtb_base[y_base, x_base]
                    
                                                 
                    for i, (dx, dy) in enumerate(offset_deltas):
                        if errors[i] < overflow_check_c:                  
                            target_x = x_base + dx
                            target_y = y_base + dy
                            
                            if (target_x >= 0 and target_x < mtb_width and 
                                target_y >= 0 and target_y < mtb_height):
                                
                                pixel_target = mtb_target[target_y, target_x]
                                diff = int(pixel_target) - int(pixel_base)
                                errors[i] += diff * diff
        
        return errors
    
    def avg_multi_function_aligned(self, image_arrays: List[np.ndarray], 
                                 offsets_x: List[int], offsets_y: List[int],
                                 width: int, height: int) -> np.ndarray:
        """
        Advanced multi-image averaging with alignment compensation and outlier rejection.
        """
        if not image_arrays:
            raise ValueError("No images to average")
        
        n_images = len(image_arrays)
        if len(image_arrays[0].shape) == 3:
            result = np.zeros((height, width, 3), dtype=np.float64)
        else:
            result = np.zeros((height, width), dtype=np.float64)
        
                                     
        for y in range(height):
            for x in range(width):
                pixel_values = []
                weights = []
                
                                                              
                for i, img_array in enumerate(image_arrays):
                                            
                    src_x = x + offsets_x[i]
                    src_y = y + offsets_y[i]
                    
                                  
                    if (src_x >= 0 and src_x < width and 
                        src_y >= 0 and src_y < height):
                        
                        pixel = img_array[src_y, src_x]
                        pixel_values.append(pixel)
                        
                                                                                       
                        offset_dist = math.sqrt(offsets_x[i]**2 + offsets_y[i]**2)
                        weight = math.exp(-offset_dist / 10.0)                     
                        weights.append(weight)
                
                if pixel_values:
                                                             
                    result[y, x] = self.robust_weighted_average(pixel_values, weights)
        
        return result.astype(np.float32)
    
    def robust_weighted_average(self, pixel_values: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """
        Compute robust weighted average with outlier rejection.
        """
        if len(pixel_values) == 1:
            return pixel_values[0]
        
        pixel_values = np.array(pixel_values)
        weights = np.array(weights)
        
        if len(pixel_values.shape) == 1:
                       
            median_val = np.median(pixel_values)
            
                                                          
            mad = np.median(np.abs(pixel_values - median_val))                             
            if mad > 0:
                z_scores = np.abs(pixel_values - median_val) / (1.4826 * mad)                  
                inlier_mask = z_scores < 2.5                                    
            else:
                inlier_mask = np.ones(len(pixel_values), dtype=bool)
            
            if np.sum(inlier_mask) > 0:
                filtered_values = pixel_values[inlier_mask]
                filtered_weights = weights[inlier_mask]
                
                                             
                return np.average(filtered_values, weights=filtered_weights)
            else:
                                                           
                return median_val
        else:
                                                
            result = np.zeros_like(pixel_values[0])
            
            for c in range(pixel_values.shape[1]):
                if len(pixel_values.shape) == 3:
                                                       
                    for c2 in range(pixel_values.shape[2]):
                        channel_values = pixel_values[:, c, c2]
                        result[c, c2] = self._robust_average_channel(channel_values, weights)
                else:
                                             
                    channel_values = pixel_values[:, c]
                    result[c] = self._robust_average_channel(channel_values, weights)
            
            return result
    
    def _robust_average_channel(self, channel_values: np.ndarray, weights: np.ndarray) -> float:
        """Helper function to compute robust average for a single channel."""
        median_val = np.median(channel_values)
        mad = np.median(np.abs(channel_values - median_val))
        
        if mad > 0:
            z_scores = np.abs(channel_values - median_val) / (1.4826 * mad)
            inlier_mask = z_scores < 2.5
        else:
            inlier_mask = np.ones(len(channel_values), dtype=bool)
        
        if np.sum(inlier_mask) > 0:
            filtered_values = channel_values[inlier_mask]
            filtered_weights = weights[inlier_mask]
            return np.average(filtered_values, weights=filtered_weights)
        else:
            return median_val
    
    def adjust_histogram_array(self, image_array: np.ndarray, width: int, height: int,
                              hdr_alpha: float, n_tiles: int, ce_preserve_blacks: bool) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        """
        if n_tiles <= 1:
            return self.global_histogram_adjustment_array(image_array, hdr_alpha, ce_preserve_blacks)
        else:
            return self.local_histogram_adjustment_array(
                image_array, hdr_alpha, n_tiles, ce_preserve_blacks, width, height
            )
    
    def global_histogram_adjustment_array(self, image_array: np.ndarray, alpha: float, 
                                        preserve_blacks: bool) -> np.ndarray:
        """Apply global histogram adjustment to image array."""
        mean_val = np.mean(image_array)
        std_val = np.std(image_array)
        
                                    
        contrast_factor = 1.0 + alpha * 0.8
        enhanced = (image_array - mean_val) * contrast_factor + mean_val
        
        if preserve_blacks:
                                     
            threshold = mean_val * 0.4
            mask = image_array < threshold
            enhanced[mask] = image_array[mask] * (1.0 + alpha * 0.2)
        
        return enhanced
    
    def local_histogram_adjustment_array(self, image_array: np.ndarray, alpha: float,
                                       n_tiles: int, preserve_blacks: bool,
                                       width: int, height: int) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        """
        tile_height = height // n_tiles
        tile_width = width // n_tiles
        result = image_array.copy()
        
                                                    
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
                                  (stop_x - start_x), (stop_y - start_y), preserve_blacks)
                
                                              
                histogram_offset = 256 * (i * n_tiles + j)
                c_histogram[histogram_offset] = histogram[0]
                for k in range(1, 256):
                    c_histogram[histogram_offset + k] = c_histogram[histogram_offset + k - 1] + histogram[k]
        
                                                  
        for y in range(height):
            for x in range(width):
                                            
                tile_x = min(int((x / width) * n_tiles), n_tiles - 1)
                tile_y = min(int((y / height) * n_tiles), n_tiles - 1)
                
                                             
                histogram_offset = 256 * (tile_y * n_tiles + tile_x)
                
                               
                if len(image_array.shape) == 3:
                                 
                    pixel = image_array[y, x]
                    enhanced_pixel = np.zeros_like(pixel)
                    
                                                            
                    value = int(np.max(pixel))
                    value = min(255, max(0, value))
                    
                                         
                    tile_start_x = int((tile_x / n_tiles) * width)
                    tile_stop_x = int(((tile_x + 1) / n_tiles) * width)
                    tile_start_y = int((tile_y / n_tiles) * height)
                    tile_stop_y = int(((tile_y + 1) / n_tiles) * height)
                    tile_size = (tile_stop_x - tile_start_x) * (tile_stop_y - tile_start_y)
                    
                    if tile_size > 0:
                        equalized_value = (c_histogram[histogram_offset + value] * 255) // tile_size
                        
                                                           
                        if value > 0:
                            scale = equalized_value / value
                            enhanced_pixel = pixel * scale
                        else:
                            enhanced_pixel = pixel
                        
                                                            
                        result[y, x] = (1.0 - alpha) * pixel + alpha * enhanced_pixel
                else:
                                     
                    value = int(image_array[y, x])
                    value = min(255, max(0, value))
                    
                    tile_start_x = int((tile_x / n_tiles) * width)
                    tile_stop_x = int(((tile_x + 1) / n_tiles) * width)
                    tile_start_y = int((tile_y / n_tiles) * height)
                    tile_stop_y = int(((tile_y + 1) / n_tiles) * height)
                    tile_size = (tile_stop_x - tile_start_x) * (tile_stop_y - tile_start_y)
                    
                    if tile_size > 0:
                        equalized_value = (c_histogram[histogram_offset + value] * 255) // tile_size
                        result[y, x] = (1.0 - alpha) * image_array[y, x] + alpha * equalized_value
        
        return result
    
    def compute_tile_histogram(self, tile: np.ndarray) -> np.ndarray:
        """
        Compute histogram for a tile based on max RGB value.
        """
        histogram = np.zeros(256, dtype=np.int32)
        
        if len(tile.shape) == 3:
                                             
            for y in range(tile.shape[0]):
                for x in range(tile.shape[1]):
                    pixel = tile[y, x]
                    value = int(np.max(pixel))
                    value = min(255, max(0, value))
                    histogram[value] += 1
        else:
                             
            for y in range(tile.shape[0]):
                for x in range(tile.shape[1]):
                    value = int(tile[y, x])
                    value = min(255, max(0, value))
                    histogram[value] += 1
        
        return histogram
    
    def clip_histogram(self, histogram: np.ndarray, temp_histogram: np.ndarray,
                      sub_width: int, sub_height: int, ce_preserve_blacks: bool):
        """
        Clip histogram for Contrast Limited AHE algorithm.
        """
        n_pixels = sub_width * sub_height
        clip_limit = (5 * n_pixels) // 256
        
                                                  
        bottom = 0
        top = clip_limit
        while top - bottom > 1:
            middle = (top + bottom) // 2
            total_clipped = sum(max(0, histogram[x] - middle) for x in range(256))
            if total_clipped > (clip_limit - middle) * 256:
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
        
        if ce_preserve_blacks:
                                          
            temp_histogram[0] = histogram[0]
            for x in range(1, 256):
                temp_histogram[x] = temp_histogram[x - 1] + histogram[x]
            
            equal_limit = n_pixels // 256
            dark_threshold_c = 128
            
            for x in range(dark_threshold_c):
                c_equal_limit = equal_limit * (x + 1)
                if temp_histogram[x] >= c_equal_limit:
                    continue
                
                alpha = 1.0 - (x / dark_threshold_c)
                limit = int(alpha * equal_limit)
                
                if histogram[x] < limit:
                                                       
                    needed = limit - histogram[x]
                    for y in range(x + 1, 256):
                        if needed <= 0:
                            break
                        if histogram[y] > equal_limit:
                            move = min(histogram[y] - equal_limit, needed)
                            histogram[x] += move
                            histogram[y] -= move
                            needed -= move
    
    def auto_alignment_simple(self, offsets_x: List[int], offsets_y: List[int], 
                            width: int, height: int, bitmaps: List[Image.Image], 
                            base_bitmap: int, use_mtb: bool, max_align_scale: int):
        """
        Simple auto-alignment wrapper function for compatibility.
        """
        self.logger.debug("autoAlignment")
        
                            
        self.offsets_x = [0] * len(bitmaps)
        self.offsets_y = [0] * len(bitmaps)
        
                                      
        result_offsets_x, result_offsets_y = self.auto_alignment_multi_image(
            bitmaps, base_bitmap, use_mtb, max_align_scale, time.time() * 1000
        )
        
                                       
        for i in range(min(len(offsets_x), len(result_offsets_x))):
            offsets_x[i] = result_offsets_x[i]
        for i in range(min(len(offsets_y), len(result_offsets_y))):
            offsets_y[i] = result_offsets_y[i]
        
                                     
        self.offsets_x = result_offsets_x
        self.offsets_y = result_offsets_y
    
    def create_mtb_images_for_alignment(self, bitmaps: List[Image.Image], 
                                      threshold: int) -> List[np.ndarray]:
        """
        Create Median Threshold Bitmaps for all images.
        """
        mtb_images = []
        
        for bitmap in bitmaps:
                                  
            gray_array = np.array(bitmap.convert('L'))
            
                                     
            mtb = np.zeros_like(gray_array, dtype=np.uint8)
            
            min_diff = 4                   
            for y in range(gray_array.shape[0]):
                for x in range(gray_array.shape[1]):
                    value = gray_array[y, x]
                    
                    if abs(value - threshold) <= min_diff:
                        mtb[y, x] = 127                           
                    elif value > threshold:
                        mtb[y, x] = 255         
                    else:
                        mtb[y, x] = 0         
            
            mtb_images.append(mtb)
        
        return mtb_images
    
    def align_images_mtb(self, bitmap0: Image.Image, bitmap1: Image.Image,
                        initial_offset_x: int = 0, initial_offset_y: int = 0,
                        max_offset: int = 8) -> Tuple[int, int]:
        """
        Align two images using MTB method with hierarchical search.
        """
                                                  
        width, height = bitmap0.size
        mtb_width = width // 2
        mtb_height = height // 2
        mtb_x = mtb_width // 2
        mtb_y = mtb_height // 2
        
        luminance0 = self.compute_median_luminance(bitmap0, mtb_x, mtb_y, mtb_width, mtb_height)
        luminance1 = self.compute_median_luminance(bitmap1, mtb_x, mtb_y, mtb_width, mtb_height)
        
                           
        mtb0 = self.create_mtb_image(bitmap0, luminance0.median_value, mtb_x, mtb_y, mtb_width, mtb_height)
        mtb1 = self.create_mtb_image(bitmap1, luminance1.median_value, mtb_x, mtb_y, mtb_width, mtb_height)
        
        if mtb0 is None or mtb1 is None:
            return initial_offset_x, initial_offset_y
        
                                        
        offset_x, offset_y = self.hierarchical_mtb_alignment(
            mtb0, mtb1, max_offset, True, mtb_width, mtb_height
        )
        
        return initial_offset_x + offset_x, initial_offset_y + offset_y
    
    def calculate_mtb_differences(self, base_mtb: np.ndarray, target_mtb: np.ndarray,
                                dx: int, dy: int) -> int:
        """
        Calculate number of different bits between MTB images with offset.
        """
        h, w = base_mtb.shape
        
                                  
        x1 = max(0, -dx)
        y1 = max(0, -dy)
        x2 = min(w, w - dx)
        y2 = min(h, h - dy)
        
        if x2 <= x1 or y2 <= y1:
            return float('inf')              
        
                                     
        base_region = base_mtb[y1:y2, x1:x2]
        target_region = target_mtb[y1+dy:y2+dy, x1+dx:x2+dx]
        
                                                                
        differences = 0
        total_pixels = 0
        
        for y in range(base_region.shape[0]):
            for x in range(base_region.shape[1]):
                base_pixel = base_region[y, x]
                target_pixel = target_region[y, x]
                
                                   
                if base_pixel != 127 and target_pixel != 127:
                    total_pixels += 1
                    if base_pixel != target_pixel:
                        differences += 1
        
        return differences if total_pixels > 0 else float('inf')
    
    def find_best_offset_correlation(self, base_image: np.ndarray, target_image: np.ndarray,
                                   max_offset: int) -> Tuple[int, int]:
        """
        Find best offset using normalized cross-correlation with multi-scale approach.
        """
        best_offset_x = 0
        best_offset_y = 0
        best_correlation = -1
        
                                           
        scales = [4, 2, 1] if max_offset > 2 else [1]
        
        for scale in scales:
            scaled_max_offset = max(1, max_offset // scale)
            
                                              
            if scale > 1:
                h, w = base_image.shape
                base_scaled = base_image[::scale, ::scale]
                target_scaled = target_image[::scale, ::scale]
            else:
                base_scaled = base_image
                target_scaled = target_image
            
                                                
            search_x = best_offset_x // scale if scale > 1 else 0
            search_y = best_offset_y // scale if scale > 1 else 0
            
            for dy in range(search_y - scaled_max_offset, search_y + scaled_max_offset + 1):
                for dx in range(search_x - scaled_max_offset, search_x + scaled_max_offset + 1):
                    correlation = self.calculate_normalized_correlation(
                        base_scaled, target_scaled, dx, dy
                    )
                    
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_offset_x = dx * scale
                        best_offset_y = dy * scale
        
        return best_offset_x, best_offset_y
    
    def calculate_normalized_correlation(self, base_image: np.ndarray, target_image: np.ndarray,
                                       dx: int, dy: int) -> float:
        """
        Calculate normalized cross-correlation between images with offset.
        """
        h, w = base_image.shape
        
                                  
        x1 = max(0, -dx)
        y1 = max(0, -dy)
        x2 = min(w, w - dx)
        y2 = min(h, h - dy)
        
        if x2 <= x1 or y2 <= y1:
            return -1              
        
                                     
        base_region = base_image[y1:y2, x1:x2]
        target_region = target_image[y1+dy:y2+dy, x1+dx:x2+dx]
        
                                                
        base_mean = np.mean(base_region)
        target_mean = np.mean(target_region)
        
        base_centered = base_region - base_mean
        target_centered = target_region - target_mean
        
        numerator = np.sum(base_centered * target_centered)
        base_norm = np.sqrt(np.sum(base_centered**2))
        target_norm = np.sqrt(np.sum(target_centered**2))
        
        if base_norm > 0 and target_norm > 0:
            return numerator / (base_norm * target_norm)
        else:
            return -1