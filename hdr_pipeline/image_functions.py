from __future__ import annotations

import math
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class TonemappingAlgorithm(Enum):
    TONEMAPALGORITHM_CLAMP = "CLAMP"
    TONEMAPALGORITHM_EXPONENTIAL = "EXPONENTIAL"
    TONEMAPALGORITHM_REINHARD = "REINHARD"
    TONEMAPALGORITHM_FU2 = "FU2"
    TONEMAPALGORITHM_ACES = "ACES"


class HistogramType(Enum):
    TYPE_RGB = "RGB"
    TYPE_LUMINANCE = "LUMINANCE"
    TYPE_VALUE = "VALUE"
    TYPE_INTENSITY = "INTENSITY"
    TYPE_LIGHTNESS = "LIGHTNESS"


class HDRProcessorException(Exception):
    UNEQUAL_SIZES = "Unequal bitmap sizes"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class CreateMTBApplyFunction:
    def __init__(self, use_mtb: bool, median_value: int):
        self.use_mtb = use_mtb
        self.median_value = median_value

    def apply_to_pixels(self, pixels: np.ndarray, width: int, height: int) -> np.ndarray:
        pixels_out = np.zeros_like(pixels, dtype=np.int32)
        if self.use_mtb:
            for i in range(len(pixels)):
                color = pixels[i]
                r = (color >> 16) & 0xFF
                g = (color >> 8) & 0xFF
                b = color & 0xFF
                value = max(r, g, b)
                diff = abs(value - self.median_value)
                if diff <= 4:
                    pixels_out[i] = 127 << 24
                elif value <= self.median_value:
                    pixels_out[i] = 0
                else:
                    pixels_out[i] = 255 << 24
        else:
            for i in range(len(pixels)):
                color = pixels[i]
                r = (color >> 16) & 0xFF
                g = (color >> 8) & 0xFF
                b = color & 0xFF
                value = max(r, g, b)
                pixels_out[i] = value << 24
        return pixels_out


class AlignMTBApplyFunction:
    def __init__(
        self,
        use_mtb: bool,
        bitmap0: np.ndarray,
        bitmap1: np.ndarray,
        offset_x: int,
        offset_y: int,
        step_size: int,
        bitmap0_width: int,
        bitmap0_height: int,
        bitmap1_width: int,
        bitmap1_height: int,
    ):
        self.use_mtb = use_mtb
        self.bitmap0 = bitmap0
        self.bitmap1 = bitmap1
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.step_size = step_size
        self.bitmap0_width = bitmap0_width
        self.bitmap0_height = bitmap0_height
        self.bitmap1_width = bitmap1_width
        self.bitmap1_height = bitmap1_height
        self.errors = np.zeros(9, dtype=np.int64)

    def apply(self, off_x: int, off_y: int, this_width: int, this_height: int):
        sy = off_y
        ey = off_y + this_height
        while sy * self.step_size + self.offset_y < self.step_size:
            sy += 1
        while (ey - 1) * self.step_size + self.offset_y >= self.bitmap1_height - self.step_size:
            ey -= 1

        if self.use_mtb:
            for cy in range(sy, ey):
                y = cy * self.step_size
                y_plus_offset = y + self.offset_y
                                                                                               
                y_plus_offset = max(0, min(y_plus_offset, self.bitmap1_height - 1))
                sx = off_x
                ex = off_x + this_width
                while sx * self.step_size + self.offset_x < self.step_size:
                    sx += 1
                while (ex - 1) * self.step_size + self.offset_x >= self.bitmap1_width - self.step_size:
                    ex -= 1

                for cx in range(sx, ex):
                    x = cx * self.step_size
                    x_plus_offset = x + self.offset_x
                                                                                               
                    x_plus_offset = max(0, min(x_plus_offset, self.bitmap1_width - 1))
                    pixel0 = (self.bitmap0[y * self.bitmap0_width + x] >> 24) & 0xFF

                                                        
                    coords = (
                        (-self.step_size, -self.step_size),
                        (0, -self.step_size),
                        (self.step_size, -self.step_size),
                        (-self.step_size, 0),
                        (0, 0),
                        (self.step_size, 0),
                        (-self.step_size, self.step_size),
                        (0, self.step_size),
                        (self.step_size, self.step_size),
                    )
                    for i, (dx, dy) in enumerate(coords):
                        py = y_plus_offset + dy
                        px = x_plus_offset + dx
                        pixel1 = (self.bitmap1[py * self.bitmap1_width + px] >> 24) & 0xFF
                        if pixel0 != pixel1 and pixel0 != 127 and pixel1 != 127:
                            self.errors[i] += 1
        else:
            overflow_check_c = 2_000_000_000
            for cy in range(sy, ey):
                y = cy * self.step_size
                y_plus_offset = y + self.offset_y
                                                                                               
                y_plus_offset = max(0, min(y_plus_offset, self.bitmap1_height - 1))
                sx = off_x
                ex = off_x + this_width
                while sx * self.step_size + self.offset_x < self.step_size:
                    sx += 1
                while (ex - 1) * self.step_size + self.offset_x >= self.bitmap1_width - self.step_size:
                    ex -= 1

                for cx in range(sx, ex):
                    x = cx * self.step_size
                    x_plus_offset = x + self.offset_x
                                                                                               
                    x_plus_offset = max(0, min(x_plus_offset, self.bitmap1_width - 1))
                    pixel0 = (self.bitmap0[y * self.bitmap0_width + x] >> 24) & 0xFF

                    coords = (
                        (-self.step_size, -self.step_size),
                        (0, -self.step_size),
                        (self.step_size, -self.step_size),
                        (-self.step_size, 0),
                        (0, 0),
                        (self.step_size, 0),
                        (-self.step_size, self.step_size),
                        (0, self.step_size),
                        (self.step_size, self.step_size),
                    )
                    for i, (dx, dy) in enumerate(coords):
                        if self.errors[i] < overflow_check_c:
                            py = y_plus_offset + dy
                            px = x_plus_offset + dx
                            pixel1 = (self.bitmap1[py * self.bitmap1_width + px] >> 24) & 0xFF
                            diff = pixel1 - pixel0
                            self.errors[i] += diff * diff

    def get_errors(self) -> np.ndarray:
        return self.errors.copy()


class HDRApplyFunction:
    def __init__(
        self,
        tonemap_algorithm: TonemappingAlgorithm,
        tonemap_scale: float,
        W: float,
        linear_scale: float,
        bitmap0: Optional[np.ndarray],
        bitmap2: Optional[np.ndarray],
        offset_x0: int,
        offset_y0: int,
        offset_x2: int,
        offset_y2: int,
        width: int,
        height: int,
        parameter_A: List[float],
        parameter_B: List[float],
    ):
        self.tonemap_algorithm = tonemap_algorithm
        self.tonemap_scale = tonemap_scale
        self.W = W
        self.linear_scale = linear_scale
        self.bitmap0 = bitmap0
        self.bitmap2 = bitmap2
        self.offset_x0 = offset_x0
        self.offset_y0 = offset_y0
        self.offset_x2 = offset_x2
        self.offset_y2 = offset_y2
        self.width = width
        self.height = height

        if len(parameter_A) != len(parameter_B):
            raise ValueError("parameter_A and parameter_B must have equal lengths")
        self.parameter_A = parameter_A.copy()
        self.parameter_B = parameter_B.copy()

    @staticmethod
    def _extract_rgb_from_int(color: int) -> Tuple[int, int, int]:
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        return r, g, b

    @staticmethod
    def _pack_color(r: int, g: int, b: int, a: int = 255) -> int:
        return (a << 24) | (r << 16) | (g << 8) | b

    def _get_bitmap_pixel(self, bitmap: np.ndarray, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return bitmap[y * self.width + x]
        return 0

    @staticmethod
    def _fu2_tonemap(x: float) -> float:
        A = 0.15
        B = 0.50
        C = 0.10
        D = 0.20
        E = 0.02
        F = 0.30
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

    def _tonemap(self, hdr_r: float, hdr_g: float, hdr_b: float) -> Tuple[int, int, int]:
        if self.tonemap_algorithm == TonemappingAlgorithm.TONEMAPALGORITHM_CLAMP:
            r = min(255, int(hdr_r + 0.5))
            g = min(255, int(hdr_g + 0.5))
            b = min(255, int(hdr_b + 0.5))
            return r, g, b
        elif self.tonemap_algorithm == TonemappingAlgorithm.TONEMAPALGORITHM_EXPONENTIAL:
            exposure = 1.2
            out_fr = self.linear_scale * 255.0 * (1.0 - math.exp(-exposure * hdr_r / 255.0))
            out_fg = self.linear_scale * 255.0 * (1.0 - math.exp(-exposure * hdr_g / 255.0))
            out_fb = self.linear_scale * 255.0 * (1.0 - math.exp(-exposure * hdr_b / 255.0))
            r = int(max(0.0, min(255.0, out_fr + 0.5)))
            g = int(max(0.0, min(255.0, out_fg + 0.5)))
            b = int(max(0.0, min(255.0, out_fb + 0.5)))
            return r, g, b
        elif self.tonemap_algorithm == TonemappingAlgorithm.TONEMAPALGORITHM_REINHARD:
            value = max(hdr_r, hdr_g, hdr_b)
            scale = 255.0 / (self.tonemap_scale + value)
            scale *= self.linear_scale
            r = int(scale * hdr_r + 0.5)
            g = int(scale * hdr_g + 0.5)
            b = int(scale * hdr_b + 0.5)
            return r, g, b
        elif self.tonemap_algorithm == TonemappingAlgorithm.TONEMAPALGORITHM_FU2:
            fu2_exposure_bias = 2.0 / 255.0
            white_scale = 255.0 / self._fu2_tonemap(self.W)
            curr_r = self._fu2_tonemap(fu2_exposure_bias * hdr_r) * white_scale
            curr_g = self._fu2_tonemap(fu2_exposure_bias * hdr_g) * white_scale
            curr_b = self._fu2_tonemap(fu2_exposure_bias * hdr_b) * white_scale
            r = int(max(0.0, min(255.0, curr_r + 0.5)))
            g = int(max(0.0, min(255.0, curr_g + 0.5)))
            b = int(max(0.0, min(255.0, curr_b + 0.5)))
            return r, g, b
        elif self.tonemap_algorithm == TonemappingAlgorithm.TONEMAPALGORITHM_ACES:
            a = 2.51
            b = 0.03
            c = 2.43
            d = 0.59
            e = 0.14
            xr = hdr_r / 255.0
            xg = hdr_g / 255.0
            xb = hdr_b / 255.0
            out_fr = 255.0 * (xr * (a * xr + b) / (xr * (c * xr + d) + e))
            out_fg = 255.0 * (xg * (a * xg + b) / (xg * (c * xg + d) + e))
            out_fb = 255.0 * (xb * (a * xb + b) / (xb * (c * xb + d) + e))
            r = int(max(0.0, min(255.0, out_fr + 0.5)))
            g = int(max(0.0, min(255.0, out_fg + 0.5)))
            b = int(max(0.0, min(255.0, out_fb + 0.5)))
            return r, g, b
        return 0, 0, 0

    def apply(self, pixels: np.ndarray, off_x: int, off_y: int, this_width: int, this_height: int) -> np.ndarray:
        pixels_out = np.zeros(this_width * this_height, dtype=np.int32)
        c = 0
        for y in range(off_y, off_y + this_height):
            for x in range(off_x, off_x + this_width):
                this_parameter_A0 = self.parameter_A[0]
                this_parameter_B0 = self.parameter_B[0]
                this_parameter_A1 = self.parameter_A[1]
                this_parameter_B1 = self.parameter_B[1]
                this_parameter_A2 = self.parameter_A[2]
                this_parameter_B2 = self.parameter_B[2]

                pixel1 = pixels[c]
                pixel1_r, pixel1_g, pixel1_b = self._extract_rgb_from_int(pixel1)

                if (
                    self.bitmap0 is not None
                    and 0 <= x + self.offset_x0 < self.width
                    and 0 <= y + self.offset_y0 < self.height
                ):
                    pixel0 = self._get_bitmap_pixel(self.bitmap0, x + self.offset_x0, y + self.offset_y0)
                    pixel0_r, pixel0_g, pixel0_b = self._extract_rgb_from_int(pixel0)
                else:
                    pixel0_r, pixel0_g, pixel0_b = pixel1_r, pixel1_g, pixel1_b
                    this_parameter_A0 = this_parameter_A1
                    this_parameter_B0 = this_parameter_B1

                if (
                    self.bitmap2 is not None
                    and 0 <= x + self.offset_x2 < self.width
                    and 0 <= y + self.offset_y2 < self.height
                ):
                    pixel2 = self._get_bitmap_pixel(self.bitmap2, x + self.offset_x2, y + self.offset_y2)
                    pixel2_r, pixel2_g, pixel2_b = self._extract_rgb_from_int(pixel2)
                else:
                    pixel2_r, pixel2_g, pixel2_b = pixel1_r, pixel1_g, pixel1_b
                    this_parameter_A2 = this_parameter_A1
                    this_parameter_B2 = this_parameter_B1

                hdr_r = 0.0
                hdr_g = 0.0
                hdr_b = 0.0
                sum_weight = 0.0

                safe_range_c = 96.0
                rgb_r = float(pixel1_r)
                rgb_g = float(pixel1_g)
                rgb_b = float(pixel1_b)
                avg = (rgb_r + rgb_g + rgb_b) / 3.0

                weight = 1.0
                if avg <= 127.5:
                    range_low_c = 32.0
                    range_high_c = 48.0
                    if avg <= range_low_c:
                        weight = 0.0
                    elif avg <= range_high_c:
                        weight = (avg - range_low_c) / (range_high_c - range_low_c)
                elif (avg - 127.5) > safe_range_c:
                    weight = 1.0 - 0.99 * ((avg - 127.5) - safe_range_c) / (127.5 - safe_range_c)

                rgb_r = this_parameter_A1 * rgb_r + this_parameter_B1
                rgb_g = this_parameter_A1 * rgb_g + this_parameter_B1
                rgb_b = this_parameter_A1 * rgb_b + this_parameter_B1

                hdr_r += weight * rgb_r
                hdr_g += weight * rgb_g
                hdr_b += weight * rgb_b
                sum_weight += weight

                if weight < 1.0:
                    base_rgb_r = rgb_r
                    base_rgb_g = rgb_g
                    base_rgb_b = rgb_b
                    weight = 1.0 - weight

                    if avg <= 127.5:
                        rgb_r = float(pixel2_r)
                        rgb_g = float(pixel2_g)
                        rgb_b = float(pixel2_b)
                        rgb_r = this_parameter_A2 * rgb_r + this_parameter_B2
                        rgb_g = this_parameter_A2 * rgb_g + this_parameter_B2
                        rgb_b = this_parameter_A2 * rgb_b + this_parameter_B2
                    else:
                        rgb_r = float(pixel0_r)
                        rgb_g = float(pixel0_g)
                        rgb_b = float(pixel0_b)
                        rgb_r = this_parameter_A0 * rgb_r + this_parameter_B0
                        rgb_g = this_parameter_A0 * rgb_g + this_parameter_B0
                        rgb_b = this_parameter_A0 * rgb_b + this_parameter_B0

                    value = max(rgb_r, rgb_g, rgb_b)
                    if value <= 250.0:
                        wiener_C_lo = 2000.0
                        wiener_C_hi = 8000.0
                        wiener_C = wiener_C_lo
                        xx = abs(value - 127.5) - 96.0
                        if xx > 0.0:
                            scale = (wiener_C_hi - wiener_C_lo) / (127.5 - 96.0)
                            wiener_C = wiener_C_lo + xx * scale
                        diff_r = base_rgb_r - rgb_r
                        diff_g = base_rgb_g - rgb_g
                        diff_b = base_rgb_b - rgb_b
                        L = diff_r * diff_r + diff_g * diff_g + diff_b * diff_b
                        ghost_weight = L / (L + wiener_C)
                        rgb_r = ghost_weight * base_rgb_r + (1.0 - ghost_weight) * rgb_r
                        rgb_g = ghost_weight * base_rgb_g + (1.0 - ghost_weight) * rgb_g
                        rgb_b = ghost_weight * base_rgb_b + (1.0 - ghost_weight) * rgb_b

                    hdr_r += weight * rgb_r
                    hdr_g += weight * rgb_g
                    hdr_b += weight * rgb_b
                    sum_weight += weight

                hdr_r /= sum_weight
                hdr_g /= sum_weight
                hdr_b /= sum_weight

                r, g, b = self._tonemap(hdr_r, hdr_g, hdr_b)
                pixels_out[c] = self._pack_color(r, g, b)
                c += 1
        return pixels_out


class RGBf_luminance:
    def __init__(self):
        self.fr = 0.0
        self.fg = 0.0
        self.fb = 0.0
        self.lum = 0.0

    def setRGB(self, fr: float, fg: float, fb: float):
        self.fr = fr
        self.fg = fg
        self.fb = fb
        self.lum = max(max(fr, fg), fb)

    def setRGB_from_pixels(self, pixels_in_rgbf: np.ndarray, x: int, y: int, width: int):
        indx = (y * width + x) * 3
        self.setRGB(pixels_in_rgbf[indx], pixels_in_rgbf[indx + 1], pixels_in_rgbf[indx + 2])


class AvgApplyFunction:
    def __init__(
        self,
        pixels_rgbf: np.ndarray,
        bitmap_new: np.ndarray,
        bitmap_orig: np.ndarray,
        width: int,
        height: int,
        offset_x_new: int,
        offset_y_new: int,
        avg_factor: float,
        wiener_C: float,
        wiener_C_cutoff: float,
    ):
        self.pixels_rgbf = pixels_rgbf
        self.bitmap_new = bitmap_new
        self.bitmap_orig = bitmap_orig
        self.width = width
        self.height = height
        self.offset_x_new = offset_x_new
        self.offset_y_new = offset_y_new
        self.avg_factor = avg_factor
        self.wiener_C = wiener_C
        self.wiener_C_cutoff = wiener_C_cutoff
        self.radius = 2

    @staticmethod
    def _extract_rgb_from_int(color: int) -> Tuple[float, float, float]:
        r = float((color >> 16) & 0xFF)
        g = float((color >> 8) & 0xFF)
        b = float(color & 0xFF)
        return r, g, b

    def _get_bitmap_pixel(self, bitmap: np.ndarray, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return bitmap[y * self.width + x]
        return 0

    def apply(self, pixels: Optional[np.ndarray], off_x: int, off_y: int, this_width: int, this_height: int):
        avg_factorp1 = self.avg_factor + 1.0
        c = 0
        for y in range(off_y, off_y + this_height):
            pixels_rgbf_indx = 3 * y * self.width + 3 * off_x
            if y + self.offset_y_new < 0 or y + self.offset_y_new >= self.height:
                if pixels is not None:
                    for _ in range(off_x, off_x + this_width):
                        color = pixels[c]
                        r, g, b = self._extract_rgb_from_int(color)
                        self.pixels_rgbf[pixels_rgbf_indx] = r
                        self.pixels_rgbf[pixels_rgbf_indx + 1] = g
                        self.pixels_rgbf[pixels_rgbf_indx + 2] = b
                        c += 1
                        pixels_rgbf_indx += 3
                continue

            y_new = y + self.offset_y_new
            for x in range(off_x, off_x + this_width):
                if pixels is not None:
                    color = pixels[c]
                    pixel_avg_fr, pixel_avg_fg, pixel_avg_fb = self._extract_rgb_from_int(color)
                    c += 1
                else:
                    pixel_avg_fr = self.pixels_rgbf[pixels_rgbf_indx]
                    pixel_avg_fg = self.pixels_rgbf[pixels_rgbf_indx + 1]
                    pixel_avg_fb = self.pixels_rgbf[pixels_rgbf_indx + 2]

                x_new = x + self.offset_x_new
                if 0 <= x_new < self.width:
                    pixel_new = self._get_bitmap_pixel(self.bitmap_new, x_new, y_new)
                    pixel_new_fr, pixel_new_fg, pixel_new_fb = self._extract_rgb_from_int(pixel_new)

                    if (
                        x - self.radius >= 0
                        and x + self.radius < self.width
                        and y - self.radius >= 0
                        and y + self.radius < self.height
                        and x_new - self.radius >= 0
                        and x_new + self.radius < self.width
                        and y_new - self.radius >= 0
                        and y_new + self.radius < self.height
                    ):
                        n_pixels_c = 5
                        sample_positions = [(-2, -2), (2, -2), (0, 0), (-2, 2), (2, 2)]
                        L = 0.0
                        for sx, sy in sample_positions:
                            pixel_orig = self._get_bitmap_pixel(self.bitmap_orig, x + sx, y + sy)
                            pixel_orig_fr, pixel_orig_fg, pixel_orig_fb = self._extract_rgb_from_int(pixel_orig)
                            if sx == 0 and sy == 0:
                                pixel_new_sample_fr = pixel_new_fr
                                pixel_new_sample_fg = pixel_new_fg
                                pixel_new_sample_fb = pixel_new_fb
                            else:
                                pixel_new_sample = self._get_bitmap_pixel(self.bitmap_new, x_new + sx, y_new + sy)
                                pixel_new_sample_fr, pixel_new_sample_fg, pixel_new_sample_fb = self._extract_rgb_from_int(
                                    pixel_new_sample
                                )
                            diff_r = pixel_orig_fr - pixel_new_sample_fr
                            diff_g = pixel_orig_fg - pixel_new_sample_fg
                            diff_b = pixel_orig_fb - pixel_new_sample_fb
                            L += diff_r * diff_r + diff_g * diff_g + diff_b * diff_b
                        L /= n_pixels_c
                    else:
                        diff_r = pixel_avg_fr - pixel_new_fr
                        diff_g = pixel_avg_fg - pixel_new_fg
                        diff_b = pixel_avg_fb - pixel_new_fb
                        L = diff_r * diff_r + diff_g * diff_g + diff_b * diff_b

                    if L <= self.wiener_C_cutoff:
                        weight = L / (L + self.wiener_C)
                        weight1 = 1.0 - weight
                        pixel_new_fr = weight * pixel_avg_fr + weight1 * pixel_new_fr
                        pixel_new_fg = weight * pixel_avg_fg + weight1 * pixel_new_fg
                        pixel_new_fb = weight * pixel_avg_fb + weight1 * pixel_new_fb
                        pixel_avg_fr = (self.avg_factor * pixel_avg_fr + pixel_new_fr) / avg_factorp1
                        pixel_avg_fg = (self.avg_factor * pixel_avg_fg + pixel_new_fg) / avg_factorp1
                        pixel_avg_fb = (self.avg_factor * pixel_avg_fb + pixel_new_fb) / avg_factorp1

                self.pixels_rgbf[pixels_rgbf_indx] = pixel_avg_fr
                self.pixels_rgbf[pixels_rgbf_indx + 1] = pixel_avg_fg
                self.pixels_rgbf[pixels_rgbf_indx + 2] = pixel_avg_fb
                pixels_rgbf_indx += 3


class AvgBrightenApplyFunction:
    def __init__(
        self,
        pixels_in_rgbf: np.ndarray,
        width: int,
        height: int,
        gain: float,
        gamma: float,
        low_x: float,
        mid_x: float,
        max_x: float,
        median_filter_strength: float,
        black_level: float,
    ):
        self.pixels_in_rgbf = pixels_in_rgbf
        self.width = width
        self.height = height
        self.median_filter_strength = median_filter_strength
        self.black_level = black_level
        self.white_level = 255.0 / (255.0 - black_level)

        self.gamma = gamma
        self.low_x = low_x
        self.mid_x = mid_x
        self.max_x = max_x

        if mid_x > low_x:
            self.gain_A = (gain * mid_x - low_x) / (mid_x - low_x)
            self.gain_B = low_x * mid_x * (1.0 - gain) / (mid_x - low_x)
        else:
            self.gain_A = 1.0
            self.gain_B = 0.0

        self.value_to_gamma_scale_lut = np.zeros(256, dtype=np.float32)
        for value in range(256):
            if value > 0:
                new_value = math.pow(value / max_x, gamma) * 255.0
                self.value_to_gamma_scale_lut[value] = new_value / value
            else:
                self.value_to_gamma_scale_lut[value] = 1.0

    def apply(self, off_x: int, off_y: int, this_width: int, this_height: int) -> np.ndarray:
        pixels_out = np.zeros(this_width * this_height, dtype=np.int32)
        rgbf_luminances = [RGBf_luminance() for _ in range(5)]

        c = 0
        for y in range(off_y, off_y + this_height):
            indx = (y * self.width + off_x) * 3
            for x in range(off_x, off_x + this_width):
                fr = self.pixels_in_rgbf[indx]
                fg = self.pixels_in_rgbf[indx + 1]
                fb = self.pixels_in_rgbf[indx + 2]
                indx += 3

                if 0 < x < self.width - 1 and 0 < y < self.height - 1:
                    rgbf_luminances[0].setRGB_from_pixels(self.pixels_in_rgbf, x, y - 1, self.width)
                    rgbf_luminances[1].setRGB_from_pixels(self.pixels_in_rgbf, x - 1, y, self.width)
                    rgbf_luminances[2].setRGB(fr, fg, fb)
                    rgbf_luminances[3].setRGB_from_pixels(self.pixels_in_rgbf, x + 1, y, self.width)
                    rgbf_luminances[4].setRGB_from_pixels(self.pixels_in_rgbf, x, y + 1, self.width)

                    if rgbf_luminances[0].lum > rgbf_luminances[1].lum:
                        rgbf_luminances[0], rgbf_luminances[1] = rgbf_luminances[1], rgbf_luminances[0]
                    if rgbf_luminances[3].lum > rgbf_luminances[4].lum:
                        rgbf_luminances[3], rgbf_luminances[4] = rgbf_luminances[4], rgbf_luminances[3]
                    if rgbf_luminances[0].lum > rgbf_luminances[3].lum:
                        rgbf_luminances[0], rgbf_luminances[3] = rgbf_luminances[3], rgbf_luminances[0]
                        rgbf_luminances[1], rgbf_luminances[4] = rgbf_luminances[4], rgbf_luminances[1]

                    if rgbf_luminances[1].lum > rgbf_luminances[2].lum:
                        if rgbf_luminances[2].lum > rgbf_luminances[3].lum:
                            if rgbf_luminances[2].lum > rgbf_luminances[4].lum:
                                rgbf_luminances[2], rgbf_luminances[4] = rgbf_luminances[4], rgbf_luminances[2]
                        else:
                            if rgbf_luminances[1].lum > rgbf_luminances[3].lum:
                                rgbf_luminances[2], rgbf_luminances[3] = rgbf_luminances[3], rgbf_luminances[2]
                            else:
                                rgbf_luminances[2], rgbf_luminances[1] = rgbf_luminances[1], rgbf_luminances[2]
                    else:
                        if rgbf_luminances[1].lum > rgbf_luminances[3].lum:
                            if rgbf_luminances[1].lum > rgbf_luminances[4].lum:
                                rgbf_luminances[2], rgbf_luminances[4] = rgbf_luminances[4], rgbf_luminances[2]
                            else:
                                rgbf_luminances[2], rgbf_luminances[1] = rgbf_luminances[1], rgbf_luminances[2]
                        else:
                            if rgbf_luminances[2].lum > rgbf_luminances[3].lum:
                                rgbf_luminances[2], rgbf_luminances[3] = rgbf_luminances[3], rgbf_luminances[2]

                    fr = (1.0 - self.median_filter_strength) * fr + self.median_filter_strength * rgbf_luminances[2].fr
                    fg = (1.0 - self.median_filter_strength) * fg + self.median_filter_strength * rgbf_luminances[2].fg
                    fb = (1.0 - self.median_filter_strength) * fb + self.median_filter_strength * rgbf_luminances[2].fb

                old_value = fg
                sum_fr = 0.0
                sum_fg = 0.0
                sum_fb = 0.0
                radius = 2
                count = 0
                sx = max(x - radius, 0)
                ex = min(x + radius, self.width - 1)
                sy = max(y - radius, 0)
                ey = min(y + radius, self.height - 1)

                for cy in range(sy, ey + 1):
                    this_indx = (cy * self.width + sx) * 3
                    for cx in range(sx, ex + 1):
                        this_fr = self.pixels_in_rgbf[this_indx]
                        this_fg = self.pixels_in_rgbf[this_indx + 1]
                        this_fb = self.pixels_in_rgbf[this_indx + 2]
                        this_indx += 3

                        this_value = this_fg
                        if this_value > 0.5:
                            scale = old_value / this_value
                            this_fr *= scale
                            this_fg *= scale
                            this_fb *= scale

                        C = 32.0
                        diff_r = fr - this_fr
                        diff_g = fg - this_fg
                        diff_b = fb - this_fb
                        L = diff_r * diff_r + diff_g * diff_g + diff_b * diff_b
                        weight = L / (L + C)

                        this_fr = this_fr + weight * diff_r
                        this_fg = this_fg + weight * diff_g
                        this_fb = this_fb + weight * diff_b

                        sum_fr += this_fr
                        sum_fg += this_fg
                        sum_fb += this_fb
                        count += 1

                fr = sum_fr / count
                fg = sum_fg / count
                fb = sum_fb / count

                if 1 <= x < self.width - 1 and 1 <= y < self.height - 1:
                    indx00 = ((y - 1) * self.width + (x - 1)) * 3
                    indx10 = ((y - 1) * self.width + x) * 3
                    indx20 = ((y - 1) * self.width + (x + 1)) * 3
                    indx01 = (y * self.width + (x - 1)) * 3
                    indx21 = (y * self.width + (x + 1)) * 3
                    indx02 = ((y + 1) * self.width + (x - 1)) * 3
                    indx12 = ((y + 1) * self.width + x) * 3
                    indx22 = ((y + 1) * self.width + (x + 1)) * 3

                    fr00 = self.pixels_in_rgbf[indx00]
                    fg00 = self.pixels_in_rgbf[indx00 + 1]
                    fb00 = self.pixels_in_rgbf[indx00 + 2]
                    fr10 = self.pixels_in_rgbf[indx10]
                    fg10 = self.pixels_in_rgbf[indx10 + 1]
                    fb10 = self.pixels_in_rgbf[indx10 + 2]
                    fr20 = self.pixels_in_rgbf[indx20]
                    fg20 = self.pixels_in_rgbf[indx20 + 1]
                    fb20 = self.pixels_in_rgbf[indx20 + 2]
                    fr01 = self.pixels_in_rgbf[indx01]
                    fg01 = self.pixels_in_rgbf[indx01 + 1]
                    fb01 = self.pixels_in_rgbf[indx01 + 2]
                    fr21 = self.pixels_in_rgbf[indx21]
                    fg21 = self.pixels_in_rgbf[indx21 + 1]
                    fb21 = self.pixels_in_rgbf[indx21 + 2]
                    fr02 = self.pixels_in_rgbf[indx02]
                    fg02 = self.pixels_in_rgbf[indx02 + 1]
                    fb02 = self.pixels_in_rgbf[indx02 + 2]
                    fr12 = self.pixels_in_rgbf[indx12]
                    fg12 = self.pixels_in_rgbf[indx12 + 1]
                    fb12 = self.pixels_in_rgbf[indx12 + 2]
                    fr22 = self.pixels_in_rgbf[indx22]
                    fg22 = self.pixels_in_rgbf[indx22 + 1]
                    fb22 = self.pixels_in_rgbf[indx22 + 2]

                    blurred_fr = (fr00 + fr10 + fr20 + fr01 + 8.0 * fr + fr21 + fr02 + fr12 + fr22) / 16.0
                    blurred_fg = (fg00 + fg10 + fg20 + fg01 + 8.0 * fg + fg21 + fg02 + fg12 + fg22) / 16.0
                    blurred_fb = (fb00 + fb10 + fb20 + fb01 + 8.0 * fb + fb21 + fb02 + fb12 + fb22) / 16.0

                    shift_fr = 1.5 * (fr - blurred_fr)
                    shift_fg = 1.5 * (fg - blurred_fg)
                    shift_fb = 1.5 * (fb - blurred_fb)

                    threshold2 = 8 * 8
                    if shift_fr * shift_fr + shift_fg * shift_fg + shift_fb * shift_fb > threshold2:
                        fr += shift_fr
                        fg += shift_fg
                        fb += shift_fb

                    fr = max(0.0, min(255.0, fr))
                    fg = max(0.0, min(255.0, fg))
                    fb = max(0.0, min(255.0, fb))

                fr = fr - self.black_level
                fg = fg - self.black_level
                fb = fb - self.black_level
                fr *= self.white_level
                fg *= self.white_level
                fb *= self.white_level
                fr = max(0.0, min(255.0, fr))
                fg = max(0.0, min(255.0, fg))
                fb = max(0.0, min(255.0, fb))

                value = max(fr, fg, fb)
                if value <= self.low_x:
                    pass
                elif value <= self.mid_x:
                    scale = self.gain_A + self.gain_B / value
                    fr *= scale
                    fg *= scale
                    fb *= scale
                else:
                    gamma_scale = self.value_to_gamma_scale_lut[int(value + 0.5)]
                    fr *= gamma_scale
                    fg *= gamma_scale
                    fb *= gamma_scale

                r = max(0, min(255, int(fr + 0.5)))
                g = max(0, min(255, int(fg + 0.5)))
                b = max(0, min(255, int(fb + 0.5)))
                pixels_out[c] = (255 << 24) | (r << 16) | (g << 8) | b
                c += 1
        return pixels_out



class DROBrightenApplyFunction:
    def __init__(self, gain: float, gamma: float, low_x: float = 0.0, mid_x: float = 127.0, max_x: float = 255.0):
                                    
        self.gamma = gamma
        self.low_x = low_x
        self.mid_x = mid_x
        self.max_x = max_x
        if mid_x > low_x:
            self.gain_A = (gain * mid_x - low_x) / (mid_x - low_x)
            self.gain_B = low_x * mid_x * (1.0 - gain) / (mid_x - low_x)
        else:
            self.gain_A = 1.0
            self.gain_B = 0.0
                                                                                
        self.value_to_gamma_scale_lut = [0.0]*256
        for value in range(256):
            new_value = (value / max_x) ** gamma * 255.0
            self.value_to_gamma_scale_lut[value] = (new_value / value) if value > 0 else 1.0

    def apply_to_int_pixels(self, pixels, width: int, height: int):
        pixels_out = [0]*len(pixels)
        for i in range(len(pixels)):
            color = pixels[i]
            r = (color >> 16) & 0xFF
            g = (color >> 8) & 0xFF
            b = color & 0xFF
            fr = float(r)
            fg = float(g)
            fb = float(b)
            value = max(fr, fg, fb)
            if value <= self.low_x:
                pass
            elif value <= self.mid_x:
                scale_factor = self.gain_A + self.gain_B / value
                fr *= scale_factor
                fg *= scale_factor
                fb *= scale_factor
            else:
                gamma_scale = self.value_to_gamma_scale_lut[int(value + 0.5)]
                fr *= gamma_scale
                fg *= gamma_scale
                fb *= gamma_scale
            r = max(0, min(255, int(fr + 0.5)))
            g = max(0, min(255, int(fg + 0.5)))
            b = max(0, min(255, int(fb + 0.5)))
            pixels_out[i] = (255 << 24) | (r << 16) | (g << 8) | b
        return np.array(pixels_out, dtype=np.int32)

    def apply_to_byte_pixels(self, pixels, off_x: int, off_y: int, this_width: int, this_height: int, width: int):
        pixels_out = np.zeros_like(pixels, dtype=np.uint8)
        c = 0
        for _y in range(off_y, off_y + this_height):
            for _x in range(off_x, off_x + this_width):
                r = int(pixels[c]); g = int(pixels[c+1]); b = int(pixels[c+2])
                if r < 0: r += 256
                if g < 0: g += 256
                if b < 0: b += 256
                fr = float(r); fg = float(g); fb = float(b)
                value = max(fr, fg, fb)
                if value <= self.low_x:
                    pass
                elif value <= self.mid_x:
                    scale_factor = self.gain_A + self.gain_B / value
                    fr *= scale_factor; fg *= scale_factor; fb *= scale_factor
                else:
                    new_value = (value / self.max_x) ** self.gamma * 255.0
                    gamma_scale = (new_value / value) if value > 0 else 1.0
                    fr *= gamma_scale; fg *= gamma_scale; fb *= gamma_scale
                r = max(0, min(255, int(fr + 0.5)))
                g = max(0, min(255, int(fg + 0.5)))
                b = max(0, min(255, int(fb + 0.5)))
                pixels_out[c] = r; pixels_out[c+1] = g; pixels_out[c+2] = b; pixels_out[c+3] = 255
                c += 4
        return pixels_out
class ComputeHistogramApplyFunction:
    def __init__(self, histogram_type: HistogramType):
        self.type = histogram_type
        self.pixels_rgb_f = None
        self.pixels_width = 0
        self.histograms = []

    def set_pixels_rgbf(self, pixels_rgb_f: np.ndarray, pixels_width: int):
        self.pixels_rgb_f = pixels_rgb_f
        self.pixels_width = pixels_width

    @staticmethod
    def _extract_rgb_from_int(color: int) -> Tuple[int, int, int]:
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        return r, g, b

    def apply_to_float_pixels(self, off_x: int, off_y: int, this_width: int, this_height: int) -> np.ndarray:
        if self.type != HistogramType.TYPE_VALUE:
            raise ValueError(f"Type not supported for float pixels: {self.type}")
        if self.pixels_rgb_f is None:
            raise ValueError("pixels_rgb_f not set - call set_pixels_rgbf() first")
        histogram = np.zeros(256, dtype=np.int32)
        for y in range(off_y, off_y + this_height):
            indx = 3 * (y * self.pixels_width + off_x)
            for _x in range(off_x, off_x + this_width):
                r = int(self.pixels_rgb_f[indx] + 0.5)
                g = int(self.pixels_rgb_f[indx + 1] + 0.5)
                b = int(self.pixels_rgb_f[indx + 2] + 0.5)
                indx += 3
                value = max(r, g, b)
                value = min(max(value, 0), 255)
                histogram[value] += 1
        return histogram

    def apply_to_int_pixels(self, pixels: np.ndarray, off_x: int, off_y: int, this_width: int, this_height: int) -> np.ndarray:
        if self.type == HistogramType.TYPE_RGB:
            histogram = np.zeros(3 * 256, dtype=np.int32)
            for c in range(this_width * this_height):
                color = pixels[c]
                histogram[(color >> 16) & 0xFF] += 1
                histogram[256 + ((color >> 8) & 0xFF)] += 1
                histogram[512 + (color & 0xFF)] += 1
        elif self.type == HistogramType.TYPE_LUMINANCE:
            histogram = np.zeros(256, dtype=np.int32)
            for c in range(this_width * this_height):
                color = pixels[c]
                fr = float((color >> 16) & 0xFF)
                fg = float((color >> 8) & 0xFF)
                fb = float(color & 0xFF)
                avg = 0.299 * fr + 0.587 * fg + 0.114 * fb
                value = min(int(avg + 0.5), 255)
                histogram[value] += 1
        elif self.type == HistogramType.TYPE_VALUE:
            histogram = np.zeros(256, dtype=np.int32)
            for c in range(this_width * this_height):
                color = pixels[c]
                value = max((color >> 16) & 0xFF, (color >> 8) & 0xFF)
                value = max(value, color & 0xFF)
                histogram[value] += 1
        elif self.type == HistogramType.TYPE_INTENSITY:
            histogram = np.zeros(256, dtype=np.int32)
            for c in range(this_width * this_height):
                color = pixels[c]
                fr = float((color >> 16) & 0xFF)
                fg = float((color >> 8) & 0xFF)
                fb = float(color & 0xFF)
                avg = (fr + fg + fb) / 3.0
                value = min(int(avg + 0.5), 255)
                histogram[value] += 1
        elif self.type == HistogramType.TYPE_LIGHTNESS:
            histogram = np.zeros(256, dtype=np.int32)
            for c in range(this_width * this_height):
                color = pixels[c]
                r = (color >> 16) & 0xFF
                g = (color >> 8) & 0xFF
                b = color & 0xFF
                max_value = max(r, g, b)
                min_value = min(r, g, b)
                avg = (min_value + max_value) / 2.0
                value = min(int(avg + 0.5), 255)
                histogram[value] += 1
        else:
            raise ValueError(f"Unknown histogram type: {self.type}")
        return histogram

    def apply_to_byte_pixels(self, pixels: np.ndarray, off_x: int, off_y: int, this_width: int, this_height: int) -> np.ndarray:
        histogram = np.zeros(256, dtype=np.int32)
        c = 0
        for _ in range(this_width * this_height):
            r = int(pixels[c])
            g = int(pixels[c + 1])
            b = int(pixels[c + 2])
            c += 4
            if r < 0:
                r += 256
            if g < 0:
                g += 256
            if b < 0:
                b += 256
            value = max(r, g, b)
            value = min(max(value, 0), 255)
            histogram[value] += 1
        return histogram


class AdjustHistogramApplyFunction:
    def __init__(self, hdr_alpha: float, n_tiles: int, width: int, height: int, c_histogram: np.ndarray):
        self.hdr_alpha = hdr_alpha
        self.n_tiles = n_tiles
        self.width = width
        self.height = height
        self.c_histogram = c_histogram

    def get_equal_value(self, histogram_offset: int, value: int) -> int:
        cdf_v = self.c_histogram[histogram_offset + value]
        cdf_0 = self.c_histogram[histogram_offset]
        n_pixels = self.c_histogram[histogram_offset + 255]
        num = float(cdf_v - cdf_0)
        den = float(n_pixels - cdf_0)
        if den == 0:
            return value
        equal_value = int(255.0 * (num / den))
        return equal_value

    def apply_to_int_pixels(self, pixels: np.ndarray, off_x: int, off_y: int, this_width: int, this_height: int) -> np.ndarray:
        pixels_out = np.zeros_like(pixels, dtype=np.int32)
        c = 0
        for y in range(off_y, off_y + this_height):
            for x in range(off_x, off_x + this_width):
                color = pixels[c]
                r = (color >> 16) & 0xFF
                g = (color >> 8) & 0xFF
                b = color & 0xFF
                value = max(r, g, b)

                tx = (float(x) * self.n_tiles) / float(self.width) - 0.5
                ty = (float(y) * self.n_tiles) / float(self.height) - 0.5
                ix = int(tx) if tx >= 0.0 else int(tx) - 1
                iy = int(ty) if ty >= 0.0 else int(ty) - 1

                if 0 <= ix < self.n_tiles - 1 and 0 <= iy < self.n_tiles - 1:
                    histogram_offset00 = 256 * (ix * self.n_tiles + iy)
                    histogram_offset10 = 256 * ((ix + 1) * self.n_tiles + iy)
                    histogram_offset01 = 256 * (ix * self.n_tiles + iy + 1)
                    histogram_offset11 = 256 * ((ix + 1) * self.n_tiles + iy + 1)
                    equal_value00 = self.get_equal_value(histogram_offset00, value)
                    equal_value10 = self.get_equal_value(histogram_offset10, value)
                    equal_value01 = self.get_equal_value(histogram_offset01, value)
                    equal_value11 = self.get_equal_value(histogram_offset11, value)
                    alpha = tx - ix
                    beta = ty - iy
                    equal_value0 = (1.0 - alpha) * equal_value00 + alpha * equal_value10
                    equal_value1 = (1.0 - alpha) * equal_value01 + alpha * equal_value11
                    equal_value = int((1.0 - beta) * equal_value0 + beta * equal_value1)
                elif 0 <= ix < self.n_tiles - 1:
                    this_y = iy + 1 if iy < 0 else iy
                    histogram_offset0 = 256 * (ix * self.n_tiles + this_y)
                    histogram_offset1 = 256 * ((ix + 1) * self.n_tiles + this_y)
                    equal_value0 = self.get_equal_value(histogram_offset0, value)
                    equal_value1 = self.get_equal_value(histogram_offset1, value)
                    beta = ty - iy
                    equal_value = int((1.0 - beta) * equal_value0 + beta * equal_value1)
                else:
                    this_x = ix + 1 if ix < 0 else ix
                    this_y = iy + 1 if iy < 0 else iy
                    histogram_offset = 256 * (this_x * self.n_tiles + this_y)
                    equal_value = self.get_equal_value(histogram_offset, value)

                new_value = int((1.0 - self.hdr_alpha) * value + self.hdr_alpha * equal_value)
                scale = float(new_value) / float(value) if value > 0 else 1.0
                r = min(255, int(r * scale + 0.5))
                g = min(255, int(g * scale + 0.5))
                b = min(255, int(b * scale + 0.5))
                pixels_out[c] = (255 << 24) | (r << 16) | (g << 8) | b
                c += 1
        return pixels_out

    def apply_to_byte_pixels(self, pixels: np.ndarray, off_x: int, off_y: int, this_width: int, this_height: int) -> np.ndarray:
        pixels_out = np.zeros_like(pixels, dtype=np.uint8)
        c = 0
        for y in range(off_y, off_y + this_height):
            for x in range(off_x, off_x + this_width):
                c = 0
                r = int(pixels[c])
                g = int(pixels[c + 1])
                b = int(pixels[c + 2])
                if r < 0:
                    r += 256
                if g < 0:
                    g += 256
                if b < 0:
                    b += 256
                value = max(r, g, b)

                tx = (float(x) * self.n_tiles) / float(self.width) - 0.5
                ty = (float(y) * self.n_tiles) / float(self.height) - 0.5
                ix = int(tx) if tx >= 0.0 else int(tx) - 1
                iy = int(ty) if ty >= 0.0 else int(ty) - 1

                if 0 <= ix < self.n_tiles - 1 and 0 <= iy < self.n_tiles - 1:
                    histogram_offset00 = 256 * (ix * self.n_tiles + iy)
                    histogram_offset10 = 256 * ((ix + 1) * self.n_tiles + iy)
                    histogram_offset01 = 256 * (ix * self.n_tiles + iy + 1)
                    histogram_offset11 = 256 * ((ix + 1) * self.n_tiles + iy + 1)
                    equal_value00 = self.get_equal_value(histogram_offset00, value)
                    equal_value10 = self.get_equal_value(histogram_offset10, value)
                    equal_value01 = self.get_equal_value(histogram_offset01, value)
                    equal_value11 = self.get_equal_value(histogram_offset11, value)
                    alpha = tx - ix
                    beta = ty - iy
                    equal_value0 = (1.0 - alpha) * equal_value00 + alpha * equal_value10
                    equal_value1 = (1.0 - alpha) * equal_value01 + alpha * equal_value11
                    equal_value = int((1.0 - beta) * equal_value0 + beta * equal_value1)
                elif 0 <= ix < self.n_tiles - 1:
                    this_y = iy + 1 if iy < 0 else iy
                    histogram_offset0 = 256 * (ix * self.n_tiles + this_y)
                    histogram_offset1 = 256 * ((ix + 1) * self.n_tiles + this_y)
                    equal_value0 = self.get_equal_value(histogram_offset0, value)
                    equal_value1 = self.get_equal_value(histogram_offset1, value)
                    alpha = tx - ix
                    equal_value = int((1.0 - alpha) * equal_value0 + alpha * equal_value1)
                elif 0 <= iy < self.n_tiles - 1:
                    this_x = ix + 1 if ix < 0 else ix
                    histogram_offset0 = 256 * (this_x * self.n_tiles + iy)
                    histogram_offset1 = 256 * (this_x * self.n_tiles + iy + 1)
                    equal_value0 = self.get_equal_value(histogram_offset0, value)
                    equal_value1 = self.get_equal_value(histogram_offset1, value)
                    beta = ty - iy
                    equal_value = int((1.0 - beta) * equal_value0 + beta * equal_value1)
                else:
                    this_x = ix + 1 if ix < 0 else ix
                    this_y = iy + 1 if iy < 0 else iy
                    histogram_offset = 256 * (this_x * self.n_tiles + this_y)
                    equal_value = self.get_equal_value(histogram_offset, value)

                new_value = int((1.0 - self.hdr_alpha) * value + self.hdr_alpha * equal_value)
                scale = float(new_value) / float(value) if value > 0 else 1.0
                pixels_out[c] = min(255, int(r * scale + 0.5))
                pixels_out[c + 1] = min(255, int(g * scale + 0.5))
                pixels_out[c + 2] = min(255, int(b * scale + 0.5))
                pixels_out[c + 3] = 255
                c += 4
        return pixels_out


class ZebraStripesApplyFunction:
    def __init__(self, zebra_stripes_threshold: int, zebra_stripes_foreground: int, zebra_stripes_background: int, zebra_stripes_width: int):
        self.zebra_stripes_threshold = zebra_stripes_threshold
        self.zebra_stripes_foreground = zebra_stripes_foreground
        self.zebra_stripes_background = zebra_stripes_background
        self.zebra_stripes_width = zebra_stripes_width

    def apply(self, pixels: np.ndarray, off_x: int, off_y: int, this_width: int, this_height: int) -> np.ndarray:
        pixels_out = np.zeros_like(pixels, dtype=np.int32)
        c = 0
        for y in range(off_y, off_y + this_height):
            for x in range(off_x, off_x + this_width):
                color = pixels[c]
                value = max((color >> 16) & 0xFF, (color >> 8) & 0xFF)
                value = max(value, color & 0xFF)
                if value >= self.zebra_stripes_threshold:
                    stripe = (x + y) // self.zebra_stripes_width
                    pixels_out[c] = self.zebra_stripes_background if (stripe % 2 == 0) else self.zebra_stripes_foreground
                else:
                    pixels_out[c] = 0
                c += 1
        return pixels_out


class FocusPeakingApplyFunction:
    def __init__(self, bitmap: np.ndarray, width: int, height: int):
        self.bitmap = bitmap
        self.width = width
        self.height = height

    def _get_bitmap_pixel(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.bitmap[y * self.width + x]
        return 0

    def apply(self, pixels: np.ndarray, off_x: int, off_y: int, this_width: int, this_height: int) -> np.ndarray:
        pixels_out = np.zeros_like(pixels, dtype=np.int32)
        c = 0
        for y in range(off_y, off_y + this_height):
            for x in range(off_x, off_x + this_width):
                strength = 0
                if 1 <= x < self.width - 1 and 1 <= y < self.height - 1:
                    pixel0c = self._get_bitmap_pixel(x - 1, y - 1)
                    pixel1c = self._get_bitmap_pixel(x, y - 1)
                    pixel2c = self._get_bitmap_pixel(x + 1, y - 1)
                    pixel3c = self._get_bitmap_pixel(x - 1, y)
                    pixel4c = pixels[c]
                    pixel5c = self._get_bitmap_pixel(x + 1, y)
                    pixel6c = self._get_bitmap_pixel(x - 1, y + 1)
                    pixel7c = self._get_bitmap_pixel(x, y + 1)
                    pixel8c = self._get_bitmap_pixel(x + 1, y + 1)

                    pixel0r = (pixel0c >> 16) & 0xFF
                    pixel0g = (pixel0c >> 8) & 0xFF
                    pixel0b = pixel0c & 0xFF
                    pixel1r = (pixel1c >> 16) & 0xFF
                    pixel1g = (pixel1c >> 8) & 0xFF
                    pixel1b = pixel1c & 0xFF
                    pixel2r = (pixel2c >> 16) & 0xFF
                    pixel2g = (pixel2c >> 8) & 0xFF
                    pixel2b = pixel2c & 0xFF
                    pixel3r = (pixel3c >> 16) & 0xFF
                    pixel3g = (pixel3c >> 8) & 0xFF
                    pixel3b = pixel3c & 0xFF
                    pixel4r = (pixel4c >> 16) & 0xFF
                    pixel4g = (pixel4c >> 8) & 0xFF
                    pixel4b = pixel4c & 0xFF
                    pixel5r = (pixel5c >> 16) & 0xFF
                    pixel5g = (pixel5c >> 8) & 0xFF
                    pixel5b = pixel5c & 0xFF
                    pixel6r = (pixel6c >> 16) & 0xFF
                    pixel6g = (pixel6c >> 8) & 0xFF
                    pixel6b = pixel6c & 0xFF
                    pixel7r = (pixel7c >> 16) & 0xFF
                    pixel7g = (pixel7c >> 8) & 0xFF
                    pixel7b = pixel7c & 0xFF
                    pixel8r = (pixel8c >> 16) & 0xFF
                    pixel8g = (pixel8c >> 8) & 0xFF
                    pixel8b = pixel8c & 0xFF

                    value_r = 8 * pixel4r - pixel0r - pixel1r - pixel2r - pixel3r - pixel5r - pixel6r - pixel7r - pixel8r
                    value_g = 8 * pixel4g - pixel0g - pixel1g - pixel2g - pixel3g - pixel5g - pixel6g - pixel7g - pixel8g
                    value_b = 8 * pixel4b - pixel0b - pixel1b - pixel2b - pixel3b - pixel5b - pixel6b - pixel7b - pixel8b
                    strength = value_r * value_r + value_g * value_g + value_b * value_b

                if strength > 256 * 256:
                    pixels_out[c] = (255 << 24) | (255 << 16) | (255 << 8) | 255
                else:
                    pixels_out[c] = 0
                c += 1
        return pixels_out


class FocusPeakingFilteredApplyFunction:
    def __init__(self, bitmap: np.ndarray, width: int, height: int):
        self.bitmap = bitmap
        self.width = width
        self.height = height

    def _get_bitmap_pixel(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.bitmap[y * self.width + x]
        return 0

    def apply(self, pixels: np.ndarray, off_x: int, off_y: int, this_width: int, this_height: int) -> np.ndarray:
        pixels_out = np.zeros_like(pixels, dtype=np.int32)
        c = 0
        for y in range(off_y, off_y + this_height):
            for x in range(off_x, off_x + this_width):
                count = 0
                if 1 <= x < self.width - 1 and 1 <= y < self.height - 1:
                    pixel1 = self._get_bitmap_pixel(x, y - 1) & 0xFF
                    pixel3 = self._get_bitmap_pixel(x - 1, y) & 0xFF
                    pixel4 = pixels[c] & 0xFF
                    pixel5 = self._get_bitmap_pixel(x + 1, y) & 0xFF
                    pixel7 = self._get_bitmap_pixel(x, y + 1) & 0xFF
                    if pixel1 == 255:
                        count += 1
                    if pixel3 == 255:
                        count += 1
                    if pixel4 == 255:
                        count += 1
                    if pixel5 == 255:
                        count += 1
                    if pixel7 == 255:
                        count += 1
                if count >= 3:
                    pixels_out[c] = (255 << 24) | (255 << 16) | (255 << 8) | 255
                else:
                    pixels_out[c] = 0
                c += 1
        return pixels_out


class ConvertToGreyscaleFunction:
    def apply(self, pixels: np.ndarray, off_x: int, off_y: int, this_width: int, this_height: int) -> np.ndarray:
        pixels_out = np.zeros_like(pixels, dtype=np.int32)
        c = 0
        for _y in range(off_y, off_y + this_height):
            for _x in range(off_x, off_x + this_width):
                color = pixels[c]
                r = (color >> 16) & 0xFF
                g = (color >> 8) & 0xFF
                b = color & 0xFF
                value = int(0.3 * float(r) + 0.59 * float(g) + 0.11 * float(b))
                pixels_out[c] = value << 24
                c += 1
        return pixels_out


pyramid_blending_weights = np.array([0.05, 0.25, 0.4, 0.25, 0.05], dtype=np.float32)


class PyramidBlendingComputeErrorFunction:
    def __init__(self, bitmap: np.ndarray, width: int):
        self.bitmap = bitmap
        self.width = width
        self.errors = []

    def apply(self, pixels: np.ndarray, off_x: int, off_y: int, this_width: int, this_height: int) -> int:
        error = 0
        overflow_check = 2_000_000_000
        c = 0
        for y in range(off_y, off_y + this_height):
            for x in range(off_x, off_x + this_width):
                color0 = pixels[c]
                r0 = (color0 >> 16) & 0xFF
                g0 = (color0 >> 8) & 0xFF
                b0 = color0 & 0xFF
                color1 = self.bitmap[y * self.width + x]
                r1 = (color1 >> 16) & 0xFF
                g1 = (color1 >> 8) & 0xFF
                b1 = color1 & 0xFF
                dr = r0 - r1
                dg = g0 - g1
                db = b0 - b1
                diff2 = dr * dr + dg * dg + db * db
                if error < overflow_check:
                    error += diff2
                c += 1
        return error


class ReduceBitmapFunction:
    def __init__(self, bitmap_in: np.ndarray, width: int, height: int):
        self.bitmap_in = bitmap_in
        self.width = width
        self.height = height

    def apply(self, off_x: int, off_y: int, this_width: int, this_height: int) -> np.ndarray:
        pixels_out = np.zeros(this_width * this_height, dtype=np.int32)
        c = 0
        for y in range(off_y, off_y + this_height):
            sy = 2 * y
            for x in range(off_x, off_x + this_width):
                sx = 2 * x
                if sx >= 2 and sx < self.width - 2 and sy >= 2 and sy < self.height - 2:
                    sum_fr = 0.0
                    sum_fg = 0.0
                    sum_fb = 0.0
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            color = self.bitmap_in[(sy + dy) * self.width + (sx + dx)]
                            r = (color >> 16) & 0xFF
                            g = (color >> 8) & 0xFF
                            b = color & 0xFF
                            fr = float(r) * pyramid_blending_weights[2 + dx] * pyramid_blending_weights[2 + dy]
                            fg = float(g) * pyramid_blending_weights[2 + dx] * pyramid_blending_weights[2 + dy]
                            fb = float(b) * pyramid_blending_weights[2 + dx] * pyramid_blending_weights[2 + dy]
                            sum_fr += fr
                            sum_fg += fg
                            sum_fb += fb
                    r = max(0, min(255, int(sum_fr + 0.5)))
                    g = max(0, min(255, int(sum_fg + 0.5)))
                    b = max(0, min(255, int(sum_fb + 0.5)))
                    pixels_out[c] = (255 << 24) | (r << 16) | (g << 8) | b
                else:
                    color = self.bitmap_in[sy * self.width + sx]
                    pixels_out[c] = color
                c += 1
        return pixels_out


def create_hdr_function(
    tonemap_algorithm: TonemappingAlgorithm,
    bitmap0: Optional[np.ndarray],
    bitmap2: Optional[np.ndarray],
    width: int,
    height: int,
    offset_x0: int = 0,
    offset_y0: int = 0,
    offset_x2: int = 0,
    offset_y2: int = 0,
    parameter_A: Optional[List[float]] = None,
    parameter_B: Optional[List[float]] = None,
    tonemap_scale: float = 255.0,
    W: float = 11.2,
    linear_scale: float = 1.0,
) -> HDRApplyFunction:
    if parameter_A is None:
        parameter_A = [1.0, 1.0, 1.0]
    if parameter_B is None:
        parameter_B = [0.0, 0.0, 0.0]
    return HDRApplyFunction(
        tonemap_algorithm,
        tonemap_scale,
        W,
        linear_scale,
        bitmap0,
        bitmap2,
        offset_x0,
        offset_y0,
        offset_x2,
        offset_y2,
        width,
        height,
        parameter_A,
        parameter_B,
    )


def create_avg_function(
    output_rgbf: np.ndarray,
    bitmap_new: np.ndarray,
    bitmap_orig: np.ndarray,
    width: int,
    height: int,
    offset_x: int,
    offset_y: int,
    avg_factor: float = 1.0,
    wiener_C: float = 10000.0,
    wiener_C_cutoff: float = 100000.0,
) -> AvgApplyFunction:
    return AvgApplyFunction(
        output_rgbf, bitmap_new, bitmap_orig, width, height, offset_x, offset_y, avg_factor, wiener_C, wiener_C_cutoff
    )


def create_dro_brighten_function(
    gain: float, gamma: float, low_x: float = 0.0, mid_x: float = 127.0, max_x: float = 255.0
) -> DROBrightenApplyFunction:
    return DROBrightenApplyFunction(gain, gamma, low_x, mid_x, max_x)


def create_histogram_function(histogram_type: HistogramType) -> ComputeHistogramApplyFunction:
    return ComputeHistogramApplyFunction(histogram_type)


def create_zebra_stripes_function(
    threshold: int = 235, foreground_color: int = 0xFF000000, background_color: int = 0x00000000, stripe_width: int = 8
) -> ZebraStripesApplyFunction:
    return ZebraStripesApplyFunction(threshold, foreground_color, background_color, stripe_width)


def create_focus_peaking_function(reference_image: np.ndarray, width: int, height: int) -> FocusPeakingApplyFunction:
    return FocusPeakingApplyFunction(reference_image, width, height)


def create_mtb_function(use_mtb: bool, median_value: int) -> CreateMTBApplyFunction:
    return CreateMTBApplyFunction(use_mtb, median_value)


def create_align_mtb_function(
    use_mtb: bool,
    bitmap0: np.ndarray,
    bitmap1: np.ndarray,
    offset_x: int,
    offset_y: int,
    step_size: int,
    bitmap0_width: int,
    bitmap0_height: int,
    bitmap1_width: int,
    bitmap1_height: int,
) -> AlignMTBApplyFunction:
    return AlignMTBApplyFunction(
        use_mtb,
        bitmap0,
        bitmap1,
        offset_x,
        offset_y,
        step_size,
        bitmap0_width,
        bitmap0_height,
        bitmap1_width,
        bitmap1_height,
    )


def create_histogram_adjustment_function(
    hdr_alpha: float, n_tiles: int, width: int, height: int, c_histogram: np.ndarray
) -> AdjustHistogramApplyFunction:
    return AdjustHistogramApplyFunction(hdr_alpha, n_tiles, width, height, c_histogram)


def compute_image_histogram(
    image: np.ndarray, histogram_type: HistogramType = HistogramType.TYPE_VALUE, width: Optional[int] = None, height: Optional[int] = None
) -> np.ndarray:
    if width is None or height is None:
        total_pixels = len(image)
        width = int(np.sqrt(total_pixels))
        height = width
        if width * height != total_pixels:
            raise ValueError("Cannot infer image dimensions - please provide width and height")
    hist_func = create_histogram_function(histogram_type)
    return hist_func.apply_to_int_pixels(image, 0, 0, width, height)
