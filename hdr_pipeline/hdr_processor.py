from __future__ import annotations
"""
HDR Core Processor (Python)
---------------------------
This module provides the core image-processing routines for your HDR pipeline.

Highlights:
- Mertens Exposure Fusion (LDR-to-LDR) with optional pre/post denoise.
- Multi-frame temporal denoise (Java-like Wiener averaging with de-ghosting).
- Linear-domain HDR merge + luminance-preserving tonemapping.
- sRGB <-> Linear helpers and zero-padded alignment.
- No SciPy dependency; OpenCV is optional (used when available).

All functions are written with defensive checks and English comments.
"""

from typing import List, Optional, Tuple, Union
import logging

import numpy as np
from PIL import Image

                                                                                 
try:
    from .utils.enums import TonemappingAlgorithm                
except Exception:
    TonemappingAlgorithm = None                             

                                                  
from .response_function import ResponseFunction

logger = logging.getLogger(__name__)


                                                                              
                 
                                                                              

def _srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    """Convert sRGB (uint8 or float in 0..255) to linear RGB in [0..1]."""
    x = arr.astype(np.float32)
    if x.max() > 1.0001:
        x = x / 255.0
    a = 0.055
    low = x <= 0.04045
    out = np.empty_like(x, dtype=np.float32)
    out[low] = x[low] / 12.92
    out[~low] = ((x[~low] + a) / (1 + a)) ** 2.4
    return np.clip(out, 0.0, 1.0)


def _linear_to_srgb(arr_lin: np.ndarray) -> np.ndarray:
    """Convert linear RGB [0..1] to sRGB uint8 [0..255]."""
    x = np.clip(arr_lin, 0.0, 1.0).astype(np.float32)
    a = 0.055
    low = x <= 0.0031308
    out = np.empty_like(x, dtype=np.float32)
    out[low] = 12.92 * x[low]
    out[~low] = (1 + a) * (x[~low] ** (1 / 2.4)) - a
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _luma_u8(img_u8: np.ndarray) -> np.ndarray:
    """Compute Rec.709 luminance from uint8 RGB."""
    f = img_u8.astype(np.float32)
    return (0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]).astype(np.float32)


def _shift_zero_pad(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Fast zero-padded shift (no wrap-around).
    Positive dx shifts right; positive dy shifts down.
    """
    h, w = arr.shape[:2]
    out = np.zeros_like(arr, dtype=arr.dtype)
    y0 = max(0, dy); y1 = min(h, h + dy)
    x0 = max(0, dx); x1 = min(w, w + dx)
    sy0 = max(0, -dy); sy1 = sy0 + (y1 - y0)
    sx0 = max(0, -dx); sx1 = sx0 + (x1 - x0)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = arr[sy0:sy1, sx0:sx1]
    return out


def _align_mtb_if_available(images: List[Image.Image]) -> List[Image.Image]:
    """
    Try to align using OpenCV AlignMTB. If OpenCV is missing, return inputs unchanged.
    """
    try:
        import cv2              
        cv_imgs = [np.array(im)[:, :, ::-1] for im in images]            
        align = cv2.createAlignMTB()
        align.process(cv_imgs, cv_imgs)
        return [Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)) for a in cv_imgs]
    except Exception as e:
        logger.debug("AlignMTB not used (%s); proceeding without alignment.", e)
        return images


                                                                              
                                                             
                                                                              

def _tonemap_luminance_preserving_lin(linear_rgb: np.ndarray, eps: float = 1e-8,
                                      curve: str = "REINHARD") -> np.ndarray:
    """
    Tonemap linear RGB *by luminance only* to preserve chroma:
      1) compute Y (luminance)
      2) Y_t = T(Y) with curve
      3) RGB' = RGB * (Y_t / (Y+eps))
    Returns linear RGB in [0..1].
    """
    R, G, B = linear_rgb[..., 0], linear_rgb[..., 1], linear_rgb[..., 2]
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

    c = curve.upper() if isinstance(curve, str) else str(curve)
    if c == "CLAMP":
        Y_t = np.clip(Y, 0.0, 1.0)
    elif c == "EXPONENTIAL":
                                            
        Y_t = 1.0 - np.exp(-Y)
    elif c in ("ACES", "FU2"):
                                
        a, b, c_, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        Y_t = np.clip((Y * (a * Y + b)) / (Y * (c_ * Y + d) + e), 0.0, 1.0)
    else:                      
        Y_t = Y / (1.0 + Y)

    scale = Y_t / (Y + eps)
    out = linear_rgb * scale[..., None]
    return np.clip(out, 0.0, 1.0)


                                                                              
                                                                        
                                                                              

def _estimate_noise_sigma_u8(img_u8: np.ndarray) -> float:
    """
    Robust noise estimate (MAD of high-frequency residual), returns sigma in 0..255.
    English: subtract a tiny blur and measure the median absolute deviation.
    """
    g = _luma_u8(img_u8)
                                         
    kern = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]], dtype=np.float32) / 9.0
    pad = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    lf = (pad[0:-2, 0:-2] * kern[0, 0] + pad[0:-2, 1:-1] * kern[0, 1] + pad[0:-2, 2:] * kern[0, 2] +
          pad[1:-1, 0:-2] * kern[1, 0] + pad[1:-1, 1:-1] * kern[1, 1] + pad[1:-1, 2:] * kern[1, 2] +
          pad[2:, 0:-2] * kern[2, 0] + pad[2:, 1:-1] * kern[2, 1] + pad[2:, 2:] * kern[2, 2])
    resid = g - lf
    mad = np.median(np.abs(resid - np.median(resid)))
                                              
    return float(1.4826 * mad)


def _pre_denoise_frame(img_u8: np.ndarray, strength: Union[str, float] = "auto") -> np.ndarray:
    """
    Low-contrast denoiser before fusion.
    - If OpenCV is available: use fastNlMeansDenoisingColored with a shadow mask.
    - Fallback: 3x3 median in shadows only.
    'strength': "auto" or float in [0..1] (0=no denoise, 1=strong).
    """
    try:
        import cv2              
        sigma = _estimate_noise_sigma_u8(img_u8)          
        if strength == "auto":
            h_lum = float(np.clip(0.8 * sigma, 3.0, 10.0))
            h_color = float(np.clip(1.2 * sigma, 4.0, 14.0))
        else:
            s = float(strength)
            h_lum = 2.0 + 10.0 * s
            h_color = 3.0 + 14.0 * s
        Y = _luma_u8(img_u8) / 255.0
        shadow_w = np.clip((0.6 - Y) / 0.6, 0.0, 1.0)                                      
        den = cv2.fastNlMeansDenoisingColored(img_u8, None,
                                              h_lum, h_color, templateWindowSize=7, searchWindowSize=21)
        out = (shadow_w[..., None] * den + (1.0 - shadow_w[..., None]) * img_u8).astype(np.uint8)
        return out
    except Exception as e:
        logger.debug("OpenCV denoise not available (%s) -> using simple fallback.", e)
                                              
        pad = np.pad(img_u8, ((1, 1), (1, 1), (0, 0)), mode="edge").astype(np.uint8)
        H, W, _ = img_u8.shape
        win = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                win.append(pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W, :])
        stack = np.stack(win, axis=0)           
        den = np.median(stack, axis=0).astype(np.uint8)
        Y = _luma_u8(img_u8) / 255.0
        shadow_w = np.clip((0.6 - Y) / 0.6, 0.0, 1.0)
        return (shadow_w[..., None] * den + (1.0 - shadow_w[..., None]) * img_u8).astype(np.uint8)


def _post_chroma_denoise(img_u8: np.ndarray, strength: float = 0.15) -> np.ndarray:
    """
    Mild chroma denoise (keeps luminance sharp). Works with/without OpenCV.
    strength: 0..1  (0=no effect)
    """
    if strength <= 1e-6:
        return img_u8
    try:
        import cv2              
        ycrcb = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        cr_d = cv2.fastNlMeansDenoising(cr, None, h=10.0 * strength,
                                        templateWindowSize=7, searchWindowSize=21)
        cb_d = cv2.fastNlMeansDenoising(cb, None, h=10.0 * strength,
                                        templateWindowSize=7, searchWindowSize=21)
        out = cv2.merge([y, cr_d, cb_d])
        out = cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)
        return out
    except Exception:
                                                      
        img = img_u8.astype(np.float32) / 255.0
        Y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        Cb = (img[..., 2] - Y) * 0.564
        Cr = (img[..., 0] - Y) * 0.713
        k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9.0

        def blur3(x: np.ndarray) -> np.ndarray:
            pad = np.pad(x, ((1, 1), (1, 1)), mode="edge")
            return (pad[0:-2, 0:-2] * k[0, 0] + pad[0:-2, 1:-1] * k[0, 1] + pad[0:-2, 2:] * k[0, 2] +
                    pad[1:-1, 0:-2] * k[1, 0] + pad[1:-1, 1:-1] * k[1, 1] + pad[1:-1, 2:] * k[1, 2] +
                    pad[2:, 0:-2] * k[2, 0] + pad[2:, 1:-1] * k[2, 1] + pad[2:, 2:] * k[2, 2])

        amt = float(np.clip(strength, 0.0, 1.0))
        Cb = (1 - amt) * Cb + amt * blur3(Cb)
        Cr = (1 - amt) * Cr + amt * blur3(Cr)
        R = Y + Cr / 0.713
        B = Y + Cb / 0.564
        G = (Y - 0.299 * R - 0.114 * B) / 0.587
        out = np.stack([R, G, B], axis=2)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)


                                                                              
            
                                                                              

class HDRCoreProcessor:
    """
    Core processor that implements HDR merge, exposure fusion and denoising.
    """

    def __init__(self,
                 enable_temporal_nr: bool = True,
                 wiener_c: float = 64.0,
                 wiener_cutoff_factor: float = 3.0,
                 nr_radius: int = 2):
        """
        enable_temporal_nr  — enable multi-frame denoise (Wiener-like).
        wiener_c            — base parameter C for w = L / (L + C).
        wiener_cutoff_factor— multiplier for L_cutoff = factor * C.
        nr_radius           — neighbourhood radius for local error (Java uses 2).
        """
        self.enable_temporal_nr = enable_temporal_nr
        self.wiener_c = float(wiener_c)
        self.wiener_cutoff_factor = float(wiener_cutoff_factor)
        self.nr_radius = int(nr_radius)

                                                                           
                                                                         
                                                                           
    def compute_response_function(self,
                                  x_samples: np.ndarray,
                                  y_samples: np.ndarray,
                                  weights: Optional[np.ndarray],
                                  func_id: int) -> ResponseFunction:
        """
        Delegate to ResponseFunction.create_from_samples to keep a single source of truth
        and avoid divergence vs. Java. If anything goes wrong, fall back to identity.
        """
        try:
            return ResponseFunction.create_from_samples(x_samples, y_samples, weights, func_id)
        except Exception as e:
            logger.error("Falling back to identity response due to: %s", e)
            return ResponseFunction(1.0, 0.0)

                                                                           
                                                                    
                                                                           
    def temporal_wiener_denoise(self,
                                bitmaps: List[Image.Image],
                                base_index: int,
                                offsets_x: List[int],
                                offsets_y: List[int],
                                parameters_a: List[float],
                                parameters_b: List[float],
                                wiener_c: Optional[float] = None,
                                wiener_c_cutoff: Optional[float] = None,
                                radius: Optional[int] = None,
                                avg_factor_init: float = 1.0) -> np.ndarray:
        """
        Multi-frame denoising using the same principle as Java's AvgApplyFunction:
          1) align each extra frame to base using integer offsets (offsets_x/y),
          2) map its brightness to the base exposure (x->A*x+B),
          3) compute local error L over 5-point neighbourhood,
          4) blend using w = L/(L+C); if L>cutoff — skip contribution (de-ghosting),
             then use running average with increasing avg_factor.
        Returns float32 array [H,W,3] in [0..255].
        """
        if radius is None:
            radius = self.nr_radius
        if wiener_c is None:
            wiener_c = self.wiener_c
        if wiener_c_cutoff is None:
            wiener_c_cutoff = self.wiener_cutoff_factor * wiener_c

        arrays = [np.array(img, dtype=np.float32) for img in bitmaps]
        h, w = arrays[base_index].shape[:2]
        base = arrays[base_index]
        avg = base.copy()
        avg_factor = float(max(1.0, avg_factor_init))

        r = int(radius)
        sample_offsets = [(-r, -r), (r, -r), (0, 0), (-r, r), (r, r)]

        for i, arr in enumerate(arrays):
            if i == base_index:
                continue
            A = float(parameters_a[i]); B = float(parameters_b[i])
            mapped = arr * A + B

            dx, dy = int(offsets_x[i]), int(offsets_y[i])
            shifted = _shift_zero_pad(mapped, dx, dy)

                                                                             
            L = np.zeros((h, w), dtype=np.float32)
            for sx, sy in sample_offsets:
                                                
                ya0 = max(0, -sy); ya1 = min(h, h - sy)
                xa0 = max(0, -sx); xa1 = min(w, w - sx)
                base_patch = base[ya0:ya1, xa0:xa1, :3]
                new_patch = shifted[ya0 + sy:ya1 + sy, xa0 + sx:xa1 + sx, :3]
                diff = base_patch - new_patch
                L[ya0:ya1, xa0:xa1] += np.sum(diff * diff, axis=2)
            L /= float(len(sample_offsets))

            weight = L / (L + float(wiener_c))                                                         
            good = (L <= float(wiener_c_cutoff))                          
            blended = weight[..., None] * avg + (1.0 - weight[..., None]) * shifted
            avg = np.where(good[..., None],
                           (avg_factor * avg + blended) / (avg_factor + 1.0),
                           avg)
            avg_factor += 1.0

        return avg

                                                                           
                                                
                                                                           
    def exposure_fusion_mertens(self, images: List[Image.Image], *, align: bool = True,
                                contrast: float = 1.0, saturation: float = 0.5, well_exposed: float = 1.5,
                                pre_denoise: bool = True, pre_strength: Union[str, float] = "auto",
                                post_chroma_denoise: bool = True, post_strength: float = 0.15) -> np.ndarray:
        """
        Exposure fusion (Mertens) to produce a pleasing LDR result directly.
        Returns: uint8 RGB ndarray.
        """
                                        
        if pre_denoise:
            images = [Image.fromarray(_pre_denoise_frame(np.array(im), strength=pre_strength))
                      for im in images]

                            
        if align:
            images = _align_mtb_if_available(images)

                                   
        try:
            import cv2              
            imgs_bgr = [np.array(im)[:, :, ::-1] for im in images]            
            merge = cv2.createMergeMertens(contrast, saturation, well_exposed)
            res = merge.process(imgs_bgr)                     
            fused = (np.clip(res, 0, 1) * 255.0 + 0.5).astype(np.uint8)
            fused = cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.debug("OpenCV MergeMertens unavailable (%s) -> using linear weighted fallback.", e)
                                              
            arrs_lin = [_srgb_to_linear(np.asarray(im, dtype=np.float32)) for im in images]
            stack = np.stack(arrs_lin, axis=0)             
            Y = 0.2126 * stack[..., 0] + 0.7152 * stack[..., 1] + 0.0722 * stack[..., 2]
            sigma = 0.25
            w = np.exp(-((Y - 0.5) ** 2) / (2 * sigma * sigma))
            w = w / (np.sum(w, axis=0, keepdims=True) + 1e-8)
            merged_lin = np.sum(stack * w[..., None], axis=0)
            fused = _linear_to_srgb(merged_lin)

        if post_chroma_denoise:
            fused = _post_chroma_denoise(fused, strength=float(post_strength))
        return fused

                                                                           
                                                                      
                                                                           
    def apply_hdr_n_function(self,
                             bitmaps: List[Image.Image],
                             base_index: int,
                             offsets_x: List[int],
                             offsets_y: List[int],
                             parameters_a: List[float],
                             parameters_b: List[float],
                             tonemapping_algorithm: Union[str, 'TonemappingAlgorithm', None] = "REINHARD",
                             tonemap_scale_c: float = 1.0,
                             linear_scale: float = 1.0,
                             use_weights: bool = True) -> Image.Image:
        """
        N-frame HDR:
          - apply exposure mapping y = a*x + b in sRGB,
          - convert to linear and merge (weighted or average),
          - luminance-preserving tonemapping,
          - convert back to sRGB.
        If enable_temporal_nr=True, the base frame is pre-cleaned with temporal Wiener denoise.
        """
        n = len(bitmaps)
        assert n == len(offsets_x) == len(offsets_y) == len(parameters_a) == len(parameters_b)

                                                               
        mapped = []
        for i in range(n):
            arr = np.asarray(bitmaps[i], dtype=np.float32)
            a = float(parameters_a[i]); b = float(parameters_b[i])
            arr = a * arr + b
            arr = _shift_zero_pad(arr, int(offsets_x[i]), int(offsets_y[i]))
            mapped.append(arr)

                                      
        if self.enable_temporal_nr and n >= 2:
            base_clean = self.temporal_wiener_denoise(
                bitmaps, base_index=base_index,
                offsets_x=offsets_x, offsets_y=offsets_y,
                parameters_a=parameters_a, parameters_b=parameters_b,
            )
            mapped[base_index] = base_clean

                               
        lin = [_srgb_to_linear(x) for x in mapped]
        stack = np.stack(lin, axis=0)             

                               
        if use_weights:
            Y = 0.2126 * stack[..., 0] + 0.7152 * stack[..., 1] + 0.0722 * stack[..., 2]
            sigma = 0.25
            w = np.exp(-((Y - 0.5) ** 2) / (2 * sigma * sigma))
            w = w / (np.sum(w, axis=0, keepdims=True) + 1e-8)
            merged_lin = np.sum(stack * w[..., None], axis=0)
        else:
            merged_lin = np.mean(stack, axis=0)

                                                  
        merged_lin = np.clip(merged_lin * float(linear_scale), 0.0, 1.0)

                                                    
        curve_name = (tonemapping_algorithm.name if (TonemappingAlgorithm and
                        isinstance(tonemapping_algorithm, TonemappingAlgorithm))
                      else str(tonemapping_algorithm or "REINHARD"))
        tm_lin = _tonemap_luminance_preserving_lin(merged_lin, curve=curve_name)

                      
        out_u8 = _linear_to_srgb(tm_lin)
        return Image.fromarray(out_u8)
