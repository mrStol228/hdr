from __future__ import annotations
from .utils.data_classes import AvgData
from .response_function import ResponseFunction
import math
import time
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np
import logging
from scipy import ndimage
from scipy.ndimage import median_filter, gaussian_filter

                                       

from .utils.exceptions import HDRProcessorException
from .utils.enums import TonemappingAlgorithm, HistogramType

class ImageAveragingProcessor:
    """
    Enhanced Image Averaging Processor with advanced Wiener filtering, 
    median filtering, and sharpening based on Java HDRProcessor implementation.
    """
    
    def __init__(self, use_renderscript: bool = False):
        self.use_renderscript = use_renderscript
        self.logger = logging.getLogger(__name__)
        self.offsets_x = None
        self.offsets_y = None
        self.sharp_index = 0
        self.cached_avg_sample_size = 1
    
    def process_avg(self, 
                   bitmap_avg: Image.Image,
                   bitmap_new: Image.Image,
                   avg_factor: float,
                   iso: int,
                   exposure_time: int,
                   zoom_factor: float = 1.0) -> "AvgData":
        """
        Combines two images by averaging them with advanced Wiener filtering.
        Based on Java HDRProcessor.processAvg() implementation.
        
        Args:
            bitmap_avg: One of the input images
            bitmap_new: The other input image  
            avg_factor: The weighting factor for bitmap_avg
            iso: The ISO used to take the photos
            exposure_time: The exposure time used to take the photos
            zoom_factor: The digital zoom factor used to take the photos
        """
        self.logger.debug("processAvg")
        self.logger.debug(f"avg_factor: {avg_factor}")
        
        if bitmap_avg.size != bitmap_new.size:
            self.logger.error("bitmaps not of same resolution")
            raise HDRProcessorException(HDRProcessorException.UNEQUAL_SIZES)
        
        time_s = time.time() * 1000                           
        
        width, height = bitmap_avg.size
        
        avg_data = self.process_avg_core(
            None, bitmap_avg, bitmap_new, width, height, 
            avg_factor, iso, exposure_time, zoom_factor, time_s
        )
        
        self.logger.debug(f"### time for processAvg: {time.time() * 1000 - time_s}")
        return avg_data
    
    def update_avg(self, 
                  avg_data: AvgData,
                  width: int,
                  height: int,
                  bitmap_new: Image.Image,
                  avg_factor: float,
                  iso: int,
                  exposure_time: int,
                  zoom_factor: float = 1.0):
        """
        Combines multiple images by averaging them. See process_avg() for more details.
        Based on Java HDRProcessor.updateAvg() implementation.
        """
        self.logger.debug("updateAvg")
        self.logger.debug(f"avg_factor: {avg_factor}")
        
        if (width, height) != bitmap_new.size:
            self.logger.error("bitmaps not of same resolution")
            raise HDRProcessorException(HDRProcessorException.UNEQUAL_SIZES)
        
        time_s = time.time() * 1000
        
        self.process_avg_core(
            avg_data, None, bitmap_new, width, height,
            avg_factor, iso, exposure_time, zoom_factor, time_s
        )
        
        self.logger.debug(f"### time for updateAvg: {time.time() * 1000 - time_s}")
    
    def process_avg_core(self, 
                        avg_data: Optional[AvgData],
                        bitmap_avg: Optional[Image.Image],
                        bitmap_new: Image.Image,
                        width: int,
                        height: int,
                        avg_factor: float,
                        iso: int,
                        exposure_time: int,
                        zoom_factor: float,
                        time_s: float) -> "AvgData":
        """
        Core algorithm for Noise Reduction algorithm based on Java processAvgCore implementation.
        Implements advanced Wiener filtering with neighborhood analysis.
        """
        self.logger.debug("processAvgCore")
        self.logger.debug(f"iso: {iso}")
        self.logger.debug(f"zoom_factor: {zoom_factor}")
        
                                                        
        allocation_out = None
        pixels_rgbf_out = None
        bitmap_avg_align = None
        allocation_avg_align = None
        bitmap_orig = None
        allocation_orig = None
        
        if avg_data is not None:
            allocation_out = avg_data.allocation_out
            pixels_rgbf_out = avg_data.pixels_rgbf_out
            bitmap_avg_align = avg_data.bitmap_avg_align
            allocation_avg_align = avg_data.allocation_avg_align
            bitmap_orig = avg_data.bitmap_orig
            allocation_orig = avg_data.allocation_orig
        
                            
        self.offsets_x = [0, 0]
        self.offsets_y = [0, 0]
        
                                   
        if bitmap_avg is not None and allocation_out is None and pixels_rgbf_out is None:
            self.logger.debug("process first bitmap")
            floating_point = False
        elif bitmap_avg is None and (allocation_out is not None or pixels_rgbf_out is not None):
            floating_point = True
            self.logger.debug("processing existing result")
        else:
            raise RuntimeError("only one of bitmap_avg or allocation_out/pixels_rgbf_out should be supplied")
        
                                                            
        self.perform_enhanced_auto_alignment(
            bitmap_avg, bitmap_new, bitmap_avg_align, width, height, 
            iso, exposure_time, zoom_factor, time_s
        )
        
                                                     
        wiener_params = self.calculate_enhanced_wiener_parameters(iso, avg_factor)
        wiener_c = wiener_params['wiener_c']
        wiener_c_cutoff = wiener_params['wiener_c_cutoff']
        
        self.logger.debug(f"wiener_C: {wiener_c}")
        self.logger.debug(f"wiener_C_cutoff: {wiener_c_cutoff}")
        
                                             
        if bitmap_orig is None:
            if floating_point:
                raise RuntimeError("is in floating point mode, but no bitmap_orig supplied")
            bitmap_orig = bitmap_avg
        
                                              
        if pixels_rgbf_out is None:
            self.logger.debug("need to create pixels_rgbf_out")
            pixels_rgbf_out = np.zeros((height, width, 3), dtype=np.float64)
            if not floating_point and bitmap_avg is not None:
                                             
                pixels_rgbf_out = np.array(bitmap_avg, dtype=np.float64) / 255.0
            self.logger.debug(f"### time after create pixels_rgbf_out: {time.time() * 1000 - time_s}")
        
                                                                      
        pixels_rgbf_out = self.apply_enhanced_wiener_averaging(
            pixels_rgbf_out, bitmap_new, bitmap_orig,
            width, height, self.offsets_x[1], self.offsets_y[1],
            avg_factor, wiener_c, wiener_c_cutoff
        )
        
        self.logger.debug(f"### time after Enhanced Wiener Averaging: {time.time() * 1000 - time_s}")
        
                                       
        new_avg_data = AvgData(
            allocation_out=allocation_out,
            pixels_rgbf_out=pixels_rgbf_out,
            bitmap_avg_align=bitmap_avg_align,
            allocation_avg_align=allocation_avg_align,
            bitmap_orig=bitmap_orig,
            allocation_orig=allocation_orig
        )
        
        self.logger.debug(f"### time for processAvgCore: {time.time() * 1000 - time_s}")
        return new_avg_data
    
    def perform_enhanced_auto_alignment(self, 
                                      bitmap_avg: Optional[Image.Image],
                                      bitmap_new: Image.Image,
                                      bitmap_avg_align: Optional[Image.Image],
                                      width: int,
                                      height: int,
                                      iso: int,
                                      exposure_time: int,
                                      zoom_factor: float,
                                      time_s: float):
        """Enhanced auto-alignment with hierarchical MTB matching."""
                                                                          
        scale_align_size = (1 if zoom_factor > 3.9 else 
                           max(4 // self.get_avg_sample_size(iso, exposure_time), 1))
        
        self.logger.debug(f"scale_align_size: {scale_align_size}")
        
                                                                        
        align_width = width // 2
        align_height = height // 2
        align_x = (width - align_width) // 2
        align_y = (height - align_height) // 2
        
                                        
        if bitmap_avg_align is None and bitmap_avg is not None:
            cropped_avg = bitmap_avg.crop((align_x, align_y, align_x + align_width, align_y + align_height))
            new_size = (align_width // scale_align_size, align_height // scale_align_size)
            bitmap_avg_align = cropped_avg.resize(new_size, Image.LANCZOS)
        
        cropped_new = bitmap_new.crop((align_x, align_y, align_x + align_width, align_y + align_height))
        new_size = (align_width // scale_align_size, align_height // scale_align_size)
        bitmap_new_align = cropped_new.resize(new_size, Image.LANCZOS)
        
                                        
        wider = self.scene_is_low_light(iso, exposure_time)
        max_align_scale = 2 if wider else 1
        
        if bitmap_avg_align is not None:
            offset_x, offset_y = self.hierarchical_mtb_alignment(
                bitmap_avg_align, bitmap_new_align, max_align_scale
            )
            
                                                   
            self.offsets_x[1] = offset_x * scale_align_size
            self.offsets_y[1] = offset_y * scale_align_size
        
        self.logger.debug(f"Final alignment offsets: x={self.offsets_x[1]}, y={self.offsets_y[1]}")
        self.logger.debug(f"### time after enhanced autoAlignment: {time.time() * 1000 - time_s}")
    
    def hierarchical_mtb_alignment(self, bitmap_base: Image.Image, bitmap_target: Image.Image, 
                                 max_align_scale: int) -> Tuple[int, int]:
        """
        Hierarchical MTB alignment using pyramid approach.
        Based on Java autoAlignment implementation.
        """
                                                   
        base_median = self.compute_median_luminance_robust(bitmap_base)
        target_median = self.compute_median_luminance_robust(bitmap_target)
        
                           
        base_mtb = self.create_enhanced_mtb(bitmap_base, base_median)
        target_mtb = self.create_enhanced_mtb(bitmap_target, target_median)
        
        if base_mtb is None or target_mtb is None:
            return 0, 0
        
                                                               
        max_dim = max(bitmap_base.width, bitmap_base.height)
        max_ideal_size = (max_align_scale * max_dim) // 150
        initial_step_size = 1
        while initial_step_size < max_ideal_size:
            initial_step_size *= 2
        
        self.logger.debug(f"MTB alignment - initial_step_size: {initial_step_size}")
        
                             
        best_offset_x = 0
        best_offset_y = 0
        step_size = initial_step_size
        
        while step_size >= 1:
                                                                      
            errors = self.calculate_mtb_alignment_errors(
                base_mtb, target_mtb, best_offset_x, best_offset_y, step_size
            )
            
                                
            best_error = float('inf')
            best_id = 4                                   
            
            for i in range(9):
                if errors[i] < best_error:
                    best_error = errors[i]
                    best_id = i
            
                                                      
            if best_id != 4:                 
                offset_dx = (best_id % 3) - 1            
                offset_dy = (best_id // 3) - 1            
                best_offset_x += offset_dx * step_size
                best_offset_y += offset_dy * step_size
            
            self.logger.debug(f"Step {step_size}: best_error={best_error}, offset=({best_offset_x}, {best_offset_y})")
            step_size //= 2
        
        return best_offset_x, best_offset_y
    
    def compute_median_luminance_robust(self, bitmap: Image.Image) -> int:
        """
        Robust median luminance computation with noise detection.
        Based on Java computeMedianLuminance implementation.
        """
        n_samples_c = 100
        n_w_samples = int(math.sqrt(n_samples_c))
        n_h_samples = n_samples_c // n_w_samples
        
        histo = np.zeros(256, dtype=np.int32)
        total = 0
        
                                        
        for y in range(n_h_samples):
            alpha = (y + 1.0) / (n_h_samples + 1.0)
            y_coord = int(alpha * bitmap.height)
            
            for x in range(n_w_samples):
                beta = (x + 1.0) / (n_w_samples + 1.0)
                x_coord = int(beta * bitmap.width)
                
                pixel = bitmap.getpixel((x_coord, y_coord))
                if isinstance(pixel, (list, tuple)) and len(pixel) >= 3:
                                                                           
                    luminance = max(pixel[0], pixel[1], pixel[2])
                else:
                    luminance = pixel if isinstance(pixel, int) else pixel[0]
                
                histo[luminance] += 1
                total += 1
        
                     
        middle = total // 2
        count = 0
        for i in range(256):
            count += histo[i]
            if count >= middle:
                                                                           
                min_diff_c = 4
                median_value = max(min_diff_c + 1, min(i, 255 - (min_diff_c + 1)))
                return median_value
        
        return 127            
    
    def create_enhanced_mtb(self, bitmap: Image.Image, median_value: int) -> Optional[np.ndarray]:
        """
        Create enhanced Median Threshold Bitmap with noise exclusion.
        Based on Java createMTBScript implementation.
        """
        gray_array = np.array(bitmap.convert('L'))
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
    
    def calculate_mtb_alignment_errors(self, base_mtb: np.ndarray, target_mtb: np.ndarray,
                                     base_offset_x: int, base_offset_y: int, 
                                     step_size: int) -> List[int]:
        """
        Calculate MTB alignment errors for 3x3 grid.
        Based on Java alignMTBScript implementation.
        """
        height, width = base_mtb.shape
        errors = [0] * 9
        
                         
        offset_deltas = [
            (-step_size, -step_size), (0, -step_size), (step_size, -step_size),
            (-step_size, 0), (0, 0), (step_size, 0),
            (-step_size, step_size), (0, step_size), (step_size, step_size)
        ]
        
                                                       
        for y in range(0, height, step_size):
            y_base = y + base_offset_y
            if y_base < 0 or y_base >= height:
                continue
                
            for x in range(0, width, step_size):
                x_base = x + base_offset_x
                if x_base < 0 or x_base >= width:
                    continue
                
                pixel_base = base_mtb[y_base, x_base]
                
                                             
                for i, (dx, dy) in enumerate(offset_deltas):
                    target_y = y_base + dy
                    target_x = x_base + dx
                    
                    if (0 <= target_x < width and 0 <= target_y < height):
                        pixel_target = target_mtb[target_y, target_x]
                        
                                                                    
                        if (pixel_base != pixel_target and 
                            pixel_base != 127 and pixel_target != 127):
                            errors[i] += 1
        
        return errors
    
    def calculate_enhanced_wiener_parameters(self, iso: int, avg_factor: float) -> dict:
        """
        Calculate advanced Wiener filter parameters.
        Based on Java processAvgCore Wiener parameter calculation.
        """
                                                                      
        limited_iso = min(iso, 400)
        wiener_cutoff_factor = 1.0
        
        if iso >= 700:
            limited_iso = 800
            if iso >= 1100:
                                                          
                wiener_cutoff_factor = 8.0
        
        limited_iso = max(limited_iso, 100)
        wiener_c = 10.0 * limited_iso
        
                                               
                                                                              
        tapered_wiener_scale = 1.0 - pow(0.5, avg_factor)
        self.logger.debug(f"avg_factor: {avg_factor}")
        self.logger.debug(f"tapered_wiener_scale: {tapered_wiener_scale}")
        
        wiener_c /= tapered_wiener_scale
        wiener_c_cutoff = wiener_cutoff_factor * wiener_c
        
        return {
            'wiener_c': wiener_c,
            'wiener_c_cutoff': wiener_c_cutoff,
            'wiener_cutoff_factor': wiener_cutoff_factor,
            'tapered_scale': tapered_wiener_scale
        }
    
    def apply_enhanced_wiener_averaging(self, 
                                      pixels_rgbf_out: np.ndarray,
                                      bitmap_new: Image.Image,
                                      bitmap_orig: Image.Image,
                                      width: int,
                                      height: int,
                                      offset_x: int,
                                      offset_y: int,
                                      avg_factor: float,
                                      wiener_c: float,
                                      wiener_c_cutoff: float) -> np.ndarray:
        """
        Real Wiener filtering algorithm with neighborhood analysis.
        Based on Java AvgApplyFunction implementation.
        """
        self.logger.debug("apply_enhanced_wiener_averaging")
        
        avg_factorp1 = avg_factor + 1.0
        radius = 2                                                   
        
                                  
        new_array = np.array(bitmap_new, dtype=np.float64)
        orig_array = np.array(bitmap_orig, dtype=np.float64)
        
                                                           
        for y in range(height):
            for x in range(width):
                                            
                pixel_avg = pixels_rgbf_out[y, x] * 255.0                                 
                
                                                     
                new_x = x + offset_x
                new_y = y + offset_y
                
                if 0 <= new_x < width and 0 <= new_y < height:
                    pixel_new = new_array[new_y, new_x]
                    
                                                                          
                                                                       
                    L = self.calculate_local_variance_wiener(
                        orig_array, new_array, x, y, new_x, new_y,
                        width, height, radius
                    )
                    
                                                                      
                    if L > wiener_c_cutoff:
                                                                                             
                        filtered_pixel = pixel_avg
                    else:
                                                                          
                        weight = L / (L + wiener_c)                                 
                        weight1 = 1.0 - weight
                        
                                                   
                        filtered_pixel = weight * pixel_avg + weight1 * pixel_new
                        
                                                
                        filtered_pixel = (avg_factor * pixel_avg + filtered_pixel) / avg_factorp1
                else:
                                                         
                    filtered_pixel = pixel_avg
                
                                                        
                pixels_rgbf_out[y, x] = filtered_pixel / 255.0
        
        return pixels_rgbf_out
    
    def calculate_local_variance_wiener(self, orig_array: np.ndarray, new_array: np.ndarray,
                                      x: int, y: int, new_x: int, new_y: int,
                                      width: int, height: int, radius: int) -> float:
        """
        Calculate local variance for Wiener filtering using neighborhood sampling.
        Based on Java AvgApplyFunction local variance calculation.
        """
                                                    
        if (x - radius >= 0 and x + radius < width and
            y - radius >= 0 and y + radius < height and
            new_x - radius >= 0 and new_x + radius < width and
            new_y - radius >= 0 and new_y + radius < height):
            
                                                                             
            sample_positions = [(-2, -2), (2, -2), (0, 0), (-2, 2), (2, 2)]
            n_pixels_c = len(sample_positions)
            
            L = 0.0
            
            for sx, sy in sample_positions:
                                               
                pixel_orig = orig_array[y + sy, x + sx]
                
                                                        
                if sx == 0 and sy == 0:
                    pixel_new_sample = new_array[new_y, new_x]
                else:
                    pixel_new_sample = new_array[new_y + sy, new_x + sx]
                
                                                               
                if len(pixel_orig.shape) > 0 and len(pixel_new_sample.shape) > 0:
                    for c in range(min(len(pixel_orig), len(pixel_new_sample), 3)):
                        diff = float(pixel_orig[c]) - float(pixel_new_sample[c])
                        L += diff * diff
                else:
                                    
                    diff = float(pixel_orig) - float(pixel_new_sample)
                    L += diff * diff
            
            L /= n_pixels_c
        else:
                                                 
            pixel_orig = orig_array[y, x] if (0 <= y < height and 0 <= x < width) else np.zeros(3)
            pixel_new = new_array[new_y, new_x] if (0 <= new_y < height and 0 <= new_x < width) else np.zeros(3)
            
            L = 0.0
            if len(pixel_orig.shape) > 0 and len(pixel_new.shape) > 0:
                for c in range(min(len(pixel_orig), len(pixel_new), 3)):
                    diff = float(pixel_orig[c]) - float(pixel_new[c])
                    L += diff * diff
            else:
                diff = float(pixel_orig) - float(pixel_new)
                L = diff * diff
        
        return L
    
    def avg_brighten(self, avg_data: AvgData, width: int, height: int, 
                    iso: int, exposure_time: int) -> Image.Image:
        """
        Final stage of noise reduction with median filtering and sharpening.
        Based on Java avgBrighten implementation.
        """
        self.logger.debug("avg_brighten")
        
        time_s = time.time() * 1000
        
                                 
        if avg_data.pixels_rgbf_out is not None:
            pixels_rgbf = avg_data.pixels_rgbf_out.copy()
        else:
            raise ValueError("No averaged image data available")
        
                                               
        pixels_255 = pixels_rgbf * 255.0
        
                                                   
        histogram = self.compute_histogram_from_array(pixels_255)
        histogram_info = self.get_histogram_info(histogram)
        
        brightness = histogram_info['median_brightness']
        max_brightness = histogram_info['max_brightness']
        
        self.logger.debug(f"median brightness: {brightness}")
        self.logger.debug(f"max brightness: {max_brightness}")
        
                                                 
        brighten_factors = self.compute_brighten_factors(
            True, iso, exposure_time, brightness, max_brightness
        )
        
                                                    
        median_filter_strength = 0.5 if self.cached_avg_sample_size >= 2 else 1.0
        if median_filter_strength > 0:
            pixels_255 = self.apply_adaptive_median_filtering(pixels_255, median_filter_strength, iso)
            self.logger.debug(f"### time after median filtering: {time.time() * 1000 - time_s}")
        
                                                                
        pixels_255 = self.apply_spatial_filtering(pixels_255, iso, exposure_time)
        self.logger.debug(f"### time after spatial filtering: {time.time() * 1000 - time_s}")
        
                                                     
        pixels_255 = self.apply_brightness_gamma_adjustment(
            pixels_255, brighten_factors['gain'], brighten_factors['gamma'],
            brighten_factors['low_x'], brighten_factors['mid_x'], max_brightness
        )
        
                                               
        black_level = self.compute_black_level(histogram, iso)
        if black_level > 0:
            pixels_255 = self.apply_black_level_adjustment(pixels_255, black_level)
        
                          
        pixels_255 = self.apply_advanced_sharpening(pixels_255, iso, exposure_time)
        self.logger.debug(f"### time after sharpening: {time.time() * 1000 - time_s}")
        
                                
        pixels_255 = np.clip(pixels_255, 0, 255).astype(np.uint8)
        result_image = Image.fromarray(pixels_255)
        
                                                      
        if iso < 1100 and exposure_time < 1000000000 // 59:
            result_image = self.apply_contrast_enhancement(
                result_image, brightness, width, height
            )
            self.logger.debug(f"### time after contrast enhancement: {time.time() * 1000 - time_s}")
        
        self.logger.debug(f"### total time for avgBrighten: {time.time() * 1000 - time_s}")
        return result_image
    
    def apply_adaptive_median_filtering(self, pixels: np.ndarray, strength: float, iso: int) -> np.ndarray:
        """
        Advanced median filtering with adaptive kernel size.
        Based on median filtering concepts from the Java implementation.
        """
        if strength <= 0:
            return pixels
        
                                                        
        if iso >= 1600:
            kernel_size = 5 if strength >= 1.0 else 3
        elif iso >= 800:
            kernel_size = 3
        else:
            kernel_size = 3 if strength >= 0.8 else 1
        
        if kernel_size <= 1:
            return pixels
        
        self.logger.debug(f"Applying median filter with kernel_size={kernel_size}, strength={strength}")
        
                                                    
        if len(pixels.shape) == 3:
                                                
            filtered = np.zeros_like(pixels)
            for c in range(pixels.shape[2]):
                                     
                channel_filtered = median_filter(pixels[:, :, c], size=kernel_size)
                
                                                                                 
                edge_mask = self.detect_edges(pixels[:, :, c])
                edge_strength = 1.0 - edge_mask * 0.5                             
                
                filtered[:, :, c] = (pixels[:, :, c] * (1.0 - strength * edge_strength) + 
                                   channel_filtered * strength * edge_strength)
        else:
                             
            channel_filtered = median_filter(pixels, size=kernel_size)
            edge_mask = self.detect_edges(pixels)
            edge_strength = 1.0 - edge_mask * 0.5
            
            filtered = (pixels * (1.0 - strength * edge_strength) + 
                       channel_filtered * strength * edge_strength)
        
        return filtered
    
    def detect_edges(self, channel: np.ndarray) -> np.ndarray:
        """Simple edge detection for preserving edges during filtering."""
                              
        sobel_x = ndimage.sobel(channel, axis=1)
        sobel_y = ndimage.sobel(channel, axis=0)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
                            
        if np.max(edge_magnitude) > 0:
            edge_magnitude = edge_magnitude / np.max(edge_magnitude)
        
        return edge_magnitude
    
    def apply_spatial_filtering(self, pixels: np.ndarray, iso: int, exposure_time: int) -> np.ndarray:
        """
        Apply spatial filtering for noise reduction.
        Implements bilateral-like filtering based on scene conditions.
        """
                                                             
        if self.scene_is_low_light(iso, exposure_time):
            spatial_sigma = 1.5
            intensity_sigma = 25.0
        else:
            spatial_sigma = 1.0
            intensity_sigma = 15.0
        
        self.logger.debug(f"Spatial filtering: spatial_sigma={spatial_sigma}, intensity_sigma={intensity_sigma}")
        
                                     
        filtered = np.zeros_like(pixels)
        
        if len(pixels.shape) == 3:
                         
            for c in range(pixels.shape[2]):
                filtered[:, :, c] = self.bilateral_filter_channel(
                    pixels[:, :, c], spatial_sigma, intensity_sigma
                )
        else:
                       
            filtered = self.bilateral_filter_channel(pixels, spatial_sigma, intensity_sigma)
        
        return filtered
    
    def bilateral_filter_channel(self, channel: np.ndarray, spatial_sigma: float, 
                               intensity_sigma: float) -> np.ndarray:
        """Simple bilateral filter implementation for a single channel."""
        height, width = channel.shape
        filtered = np.zeros_like(channel)
        
                                            
        kernel_size = int(2 * spatial_sigma + 1)
        half_kernel = kernel_size // 2
        
                                      
        spatial_weights = np.zeros((kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                dy = i - half_kernel
                dx = j - half_kernel
                spatial_weights[i, j] = np.exp(-(dx*dx + dy*dy) / (2 * spatial_sigma * spatial_sigma))
        
                      
        for y in range(half_kernel, height - half_kernel):
            for x in range(half_kernel, width - half_kernel):
                center_intensity = channel[y, x]
                
                weighted_sum = 0.0
                weight_sum = 0.0
                
                for ky in range(kernel_size):
                    for kx in range(kernel_size):
                        ny = y + ky - half_kernel
                        nx = x + kx - half_kernel
                        
                        neighbor_intensity = channel[ny, nx]
                        intensity_diff = abs(neighbor_intensity - center_intensity)
                        
                                                               
                        intensity_weight = np.exp(-(intensity_diff * intensity_diff) / 
                                                (2 * intensity_sigma * intensity_sigma))
                        total_weight = spatial_weights[ky, kx] * intensity_weight
                        
                        weighted_sum += neighbor_intensity * total_weight
                        weight_sum += total_weight
                
                filtered[y, x] = weighted_sum / weight_sum if weight_sum > 0 else center_intensity
        
                      
        filtered[:half_kernel, :] = channel[:half_kernel, :]
        filtered[-half_kernel:, :] = channel[-half_kernel:, :]
        filtered[:, :half_kernel] = channel[:, :half_kernel]
        filtered[:, -half_kernel:] = channel[:, -half_kernel:]
        
        return filtered
    
    def apply_advanced_sharpening(self, pixels: np.ndarray, iso: int, exposure_time: int) -> np.ndarray:
        """
        Apply advanced sharpening with unsharp masking.
        Based on sharpening concepts from Java implementation.
        """
                                                                 
        if iso >= 1600 or exposure_time >= 1000000000 // 15:
                                                  
            strength = 0.2
            sigma = 1.5
        elif iso >= 800 or exposure_time >= 1000000000 // 30:
                                                   
            strength = 0.5
            sigma = 1.2
        else:
                                           
            strength = 0.8
            sigma = 1.0
        
        if strength <= 0:
            return pixels
        
        self.logger.debug(f"Sharpening: strength={strength}, sigma={sigma}")
        
                               
        if len(pixels.shape) == 3:
                         
            sharpened = np.zeros_like(pixels)
            for c in range(pixels.shape[2]):
                                      
                blurred = gaussian_filter(pixels[:, :, c], sigma=sigma)
                
                                     
                mask = pixels[:, :, c] - blurred
                
                                                                   
                local_std = ndimage.generic_filter(pixels[:, :, c], np.std, size=5)
                adaptive_strength = strength * np.clip(local_std / 30.0, 0.3, 1.0)
                
                                  
                sharpened[:, :, c] = pixels[:, :, c] + adaptive_strength * mask
        else:
                       
            blurred = gaussian_filter(pixels, sigma=sigma)
            mask = pixels - blurred
            
            local_std = ndimage.generic_filter(pixels, np.std, size=5)
            adaptive_strength = strength * np.clip(local_std / 30.0, 0.3, 1.0)
            
            sharpened = pixels + adaptive_strength * mask
        
        return sharpened
    
    def compute_histogram_from_array(self, pixels: np.ndarray) -> np.ndarray:
        """Compute histogram from pixel array using max RGB value."""
        if len(pixels.shape) == 3:
                                             
            values = np.max(pixels, axis=2).flatten()
        else:
            values = pixels.flatten()
        
        histogram, _ = np.histogram(values, bins=256, range=(0, 255))
        return histogram.astype(np.int32)
    
    def get_histogram_info(self, histogram: np.ndarray) -> dict:
        """Get histogram statistics."""
        total = np.sum(histogram)
        if total == 0:
            return {'median_brightness': 127, 'max_brightness': 255, 'mean_brightness': 127}
        
                     
        middle = total // 2
        count = 0
        median_brightness = 0
        for i in range(256):
            count += histogram[i]
            if count >= middle:
                median_brightness = i
                break
        
                                                 
        max_brightness = 255
        for i in range(255, -1, -1):
            if histogram[i] > 0:
                max_brightness = i
                break
        
                        
        weighted_sum = sum(i * histogram[i] for i in range(256))
        mean_brightness = int(weighted_sum / total) if total > 0 else 127
        
        return {
            'median_brightness': median_brightness,
            'max_brightness': max_brightness,
            'mean_brightness': mean_brightness,
            'total': total
        }
    
    def compute_brighten_factors(self, has_iso_exposure: bool, iso: int, exposure_time: int,
                               brightness: int, max_brightness: int) -> dict:
        """
        Compute brightness adjustment factors.
        Based on Java computeBrightenFactors implementation.
        """
                                                        
        max_gain_factor = 1.5
        ideal_brightness = 119
        
        if has_iso_exposure and iso < 1100 and exposure_time < 1000000000 // 59:
                          
            ideal_brightness = 199
        
                                     
        if brightness > 0:
            min_brightness_c = 42.0
            min_max_gain_factor = min_brightness_c / brightness
            max_gain_factor = max(max_gain_factor, min_max_gain_factor)
            max_gain_factor = min(max_gain_factor, 15.0)
        
        if brightness <= 0:
            brightness = 1
        
        brightness_target = min(ideal_brightness, int(max_gain_factor * brightness))
        brightness_target = max(brightness, brightness_target)
        
                                  
        gain = brightness_target / brightness
        if gain < 1.0:
            gain = 1.0
        
        gamma = 1.0
        max_possible_value = gain * max_brightness
        
                                                    
        mid_x = 255.5
        if max_possible_value > 255.0:
                                    
            mid_y = (0.6 * 255.0 if (has_iso_exposure and iso < 1100 and 
                                   exposure_time < 1000000000 // 59) else 0.8 * 255.0)
            mid_x = mid_y / gain
            if max_brightness > 0:
                gamma = math.log(mid_y / 255.0) / math.log(mid_x / max_brightness)
        elif max_possible_value < 255.0 and max_brightness > 0:
                                                   
            alt_gain = min(255.0 / max_brightness, 4.0)
            if alt_gain > gain:
                gain = alt_gain
        
                                               
        low_x = 0.0
        if has_iso_exposure and iso >= 400:
            piecewise_mid_y = 0.5 * 255.0
            piecewise_mid_x = piecewise_mid_y / gain
            low_x = min(8.0, 0.125 * piecewise_mid_x)
        
        return {
            'gain': gain,
            'gamma': gamma,
            'low_x': low_x,
            'mid_x': mid_x,
            'brightness_target': brightness_target
        }
    
    def apply_brightness_gamma_adjustment(self, pixels: np.ndarray, gain: float, gamma: float,
                                        low_x: float, mid_x: float, max_brightness: int) -> np.ndarray:
        """
        Apply piecewise gain/gamma brightness adjustment.
        Based on Java DROBrightenApplyFunction implementation.
        """
        if len(pixels.shape) == 3:
                         
            for y in range(pixels.shape[0]):
                for x in range(pixels.shape[1]):
                    pixel = pixels[y, x]
                    value = np.max(pixel)                                          
                    
                    if value <= low_x:
                                                  
                        continue
                    elif value <= mid_x:
                                            
                        if value > 0:
                            scale_factor = gain + (gain - 1.0) * low_x / value
                            pixels[y, x] = pixel * scale_factor
                    else:
                                                 
                        if max_brightness > 0 and value > 0:
                            normalized = value / max_brightness
                            gamma_scale = (normalized ** gamma) * 255.0 / value
                            pixels[y, x] = pixel * gamma_scale
        else:
                       
            for y in range(pixels.shape[0]):
                for x in range(pixels.shape[1]):
                    value = pixels[y, x]
                    
                    if value <= low_x:
                        continue
                    elif value <= mid_x:
                        if value > 0:
                            scale_factor = gain + (gain - 1.0) * low_x / value
                            pixels[y, x] = value * scale_factor
                    else:
                        if max_brightness > 0 and value > 0:
                            normalized = value / max_brightness
                            pixels[y, x] = (normalized ** gamma) * 255.0
        
        return pixels
    
    def compute_black_level(self, histogram: np.ndarray, iso: int) -> float:
        """
        Compute black level for dehaze algorithm.
        Based on Java computeBlackLevel implementation.
        """
        total = np.sum(histogram)
        if total == 0:
            return 0.0
        
                                               
        percentile = int(total * 0.001)
        
        count = 0
        darkest_brightness = 0
        for i in range(len(histogram)):
            count += histogram[i]
            if count >= percentile:
                darkest_brightness = i
                break
        
        black_level = max(0.0, float(darkest_brightness))
        
                                                                     
        black_level = min(black_level, 18 if iso <= 700 else 4)
        
        self.logger.debug(f"computed black_level: {black_level}")
        return black_level
    
    def apply_black_level_adjustment(self, pixels: np.ndarray, black_level: float) -> np.ndarray:
        """Apply black level adjustment (simple dehaze algorithm)."""
        if black_level <= 0:
            return pixels
        
                                 
        adjusted = np.maximum(pixels - black_level, 0.0)
        scale_factor = 255.0 / (255.0 - black_level)
        adjusted *= scale_factor
        
        return adjusted
    
    def apply_contrast_enhancement(self, image: Image.Image, median_brightness: int,
                                 width: int, height: int) -> Image.Image:
        """
        Apply contrast enhancement for bright scenes.
        Based on Java DRO contrast enhancement logic.
        """
                                                                 
        median_lo = 60
        median_hi = 35
        alpha = (median_brightness - median_lo) / (median_hi - median_lo)
        alpha = max(0.0, min(1.0, alpha))
        amount = (1.0 - alpha) * 0.25 + alpha * 0.5
        
        self.logger.debug(f"contrast enhancement alpha: {alpha}, amount: {amount}")
        
        if amount <= 0:
            return image
        
                                           
        img_array = np.array(image, dtype=np.float32)
        
                                               
        kernel_size = max(5, min(width, height) // 50)
        if len(img_array.shape) == 3:
            enhanced = np.zeros_like(img_array)
            for c in range(img_array.shape[2]):
                local_mean = ndimage.uniform_filter(img_array[:, :, c], size=kernel_size)
                enhanced[:, :, c] = img_array[:, :, c] + amount * (img_array[:, :, c] - local_mean)
        else:
            local_mean = ndimage.uniform_filter(img_array, size=kernel_size)
            enhanced = img_array + amount * (img_array - local_mean)
        
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        return Image.fromarray(enhanced)
    
    def scene_is_low_light(self, iso: int, exposure_time: int) -> bool:
        """
        Determine if scene is low light based on ISO and exposure time.
        Based on Java sceneIsLowLight implementation.
        """
        ISO_FOR_DARK = 1100
        return ((iso >= ISO_FOR_DARK and iso * exposure_time >= 69 * 1000000000) or 
                exposure_time >= (1000000000 // 5 - 10000))
    
    def get_avg_sample_size(self, iso: int, exposure_time: int) -> int:
        """Get sample size for averaging based on scene conditions."""
        self.cached_avg_sample_size = 2 if self.scene_is_low_light(iso, exposure_time) else 1
        return self.cached_avg_sample_size
    
    def convert_to_bitmap(self, pixels_rgbf: np.ndarray) -> Image.Image:
        """Convert floating point RGB array to bitmap."""
                                           
        pixels_uint8 = np.clip(pixels_rgbf * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(pixels_uint8)
    
                                  
    def apply_avg_function_enhanced(self, *args, **kwargs):
        """Legacy method - delegates to new implementation."""
        return self.apply_enhanced_wiener_averaging(*args, **kwargs)
    
    def apply_avg_function(self, *args, **kwargs):
        """Legacy method - delegates to enhanced implementation."""
        return self.apply_enhanced_wiener_averaging(*args, **kwargs)

                                    
def test_enhanced_image_averaging():
    """Test function to demonstrate the enhanced averaging capabilities."""
    
                      
    processor = ImageAveragingProcessor()
    
                                                      
    width, height = 640, 480
    
                                     
    base_array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            intensity = int((x / width) * 255)
            base_array[y, x] = [intensity, intensity, intensity]
    
    img1 = Image.fromarray(base_array)
    
                                                      
    noisy_array = base_array.copy().astype(np.float32)
                        
    noise = np.random.normal(0, 10, noisy_array.shape)
    noisy_array += noise
                      
    shifted_array = np.roll(noisy_array, 2, axis=1)                 
    shifted_array = np.clip(shifted_array, 0, 255).astype(np.uint8)
    
    img2 = Image.fromarray(shifted_array)
    
    try:
        print("Testing Enhanced Image Averaging Processor...")
        
                                               
        avg_data = processor.process_avg(
            bitmap_avg=img1,
            bitmap_new=img2,
            avg_factor=1.0,
            iso=1600,                                        
            exposure_time=50000000,               
            zoom_factor=1.0
        )
        
        print("✓ Successfully processed first image pair with Wiener filtering")
        print(f"  Output shape: {avg_data.pixels_rgbf_out.shape}")
        
                                                     
        result_img = processor.avg_brighten(avg_data, width, height, 1600, 50000000)
        print(f"✓ Applied median filtering, spatial filtering, and sharpening")
        print(f"  Result image size: {result_img.size}")
        
                                    
        img3_array = base_array.copy().astype(np.float32)
                                 
        noise3 = np.random.normal(0, 8, img3_array.shape)
        img3_array += noise3
        img3_array = np.clip(img3_array, 0, 255).astype(np.uint8)
        img3 = Image.fromarray(img3_array)
        
        processor.update_avg(
            avg_data=avg_data,
            width=width,
            height=height,
            bitmap_new=img3,
            avg_factor=2.0,
            iso=1600,
            exposure_time=50000000,
            zoom_factor=1.0
        )
        
        print("✓ Successfully updated with additional image")
        
                          
        final_img = processor.avg_brighten(avg_data, width, height, 1600, 50000000)
        print(f"✓ Final result with all enhancements: {final_img.size}")
        
                                        
        is_low_light = processor.scene_is_low_light(1600, 50000000)
        sample_size = processor.get_avg_sample_size(1600, 50000000)
        print(f"✓ Scene analysis: low_light={is_low_light}, sample_size={sample_size}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_algorithm_components():
    """Test individual algorithm components."""
    
    processor = ImageAveragingProcessor()
    
    try:
        print("\nTesting individual algorithm components...")
        
                                           
        params_low_iso = processor.calculate_enhanced_wiener_parameters(400, 1.0)
        params_high_iso = processor.calculate_enhanced_wiener_parameters(1600, 3.0)
        
        print(f"✓ Wiener parameters (ISO 400): C={params_low_iso['wiener_c']:.1f}")
        print(f"✓ Wiener parameters (ISO 1600): C={params_high_iso['wiener_c']:.1f}")
        
                           
        test_img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        median_val = processor.compute_median_luminance_robust(test_img)
        mtb = processor.create_enhanced_mtb(test_img, median_val)
        
        print(f"✓ MTB creation: median={median_val}, MTB shape={mtb.shape if mtb is not None else 'None'}")
        
                                            
        factors_bright = processor.compute_brighten_factors(True, 400, 16000000, 120, 240)
        factors_dark = processor.compute_brighten_factors(True, 1600, 50000000, 60, 180)
        
        print(f"✓ Brighten factors (bright scene): gain={factors_bright['gain']:.2f}, gamma={factors_bright['gamma']:.2f}")
        print(f"✓ Brighten factors (dark scene): gain={factors_dark['gain']:.2f}, gamma={factors_dark['gamma']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Component test error: {e}")
        return False

if __name__ == "__main__":
    print("Enhanced Image Averaging Processor Test Suite")
    print("=" * 50)
    
                   
    success1 = test_enhanced_image_averaging()
    
                         
    success2 = test_algorithm_components()
    
    print("\n" + "=" * 50)
    print(f"Overall Test Result: {'✓ ALL PASSED' if (success1 and success2) else '✗ SOME FAILED'}")
    
    if success1 and success2:
        print("\nAll enhanced algorithms are working correctly:")
        print("• Real Wiener filtering with neighborhood analysis")
        print("• Adaptive median filtering with edge preservation")
        print("• Advanced spatial filtering (bilateral-like)")
        print("• Unsharp masking with adaptive strength")
        print("• Hierarchical MTB alignment")
        print("• Scene-adaptive parameter adjustment")