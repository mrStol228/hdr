                      
                       
"""
demo_hdr.py
-----------
A self-contained demo to load 4 images with different ISO/exposures, visualize
intermediate steps (alignment, MTB masks, histograms, response fits), and produce
a tonemapped HDR result.

It will try to import your refactored HDR package (ResponseFunction, HDRCoreProcessor).
If that import fails, it falls back to local minimal implementations for the demo.

Usage:
    python demo_hdr.py --images img1 img2 img3 img4 --out out_dir [--tonemap REINHARD|CLAMP|EXPONENTIAL|FU2|ACES]
"""

import os
import sys
import argparse
import logging
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

                                                  
                                
                                      
                                        
import matplotlib.pyplot as plt

                                                                      
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

                                                                                    
                                                      
                                                                             
PKG_TRY_DONE = False
ResponseFunction = None
HDRCoreProcessor = None
TonemapEnum = None

                                              


def exposure_fusion_mertens(images, contrast=1.0, saturation=0.5, well_exposed=1.5):
    """
    Exposure fusion (Mertens) using OpenCV if available; returns uint8 RGB.
    This avoids building an HDR radiance map and directly fuses into a nice LDR result.
    """
    try:
        import cv2
        imgs_bgr = [cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR) for im in images]
        merge = cv2.createMergeMertens(contrast, saturation, well_exposed)
        res = merge.process(imgs_bgr)                     
        res_u8 = (np.clip(res, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        res_rgb = cv2.cvtColor(res_u8, cv2.COLOR_BGR2RGB)
        return res_rgb
    except Exception:
                                                                          
                                                                               
        def srgb_to_linear(arr):
            x = arr.astype(np.float32) / 255.0
            a = 0.055
            out = np.empty_like(x)
            low = x <= 0.04045
            out[low]  = x[low] / 12.92
            out[~low] = ((x[~low] + a) / (1 + a)) ** 2.4
            return np.clip(out, 0.0, 1.0)
        arrs = [srgb_to_linear(np.asarray(im, dtype=np.float32)) for im in images]
        stack = np.stack(arrs, axis=0)             
        Y = 0.2126*stack[...,0] + 0.7152*stack[...,1] + 0.0722*stack[...,2]
                                            
        sigma = 0.25
        w = np.exp(-((Y - 0.5)**2) / (2*sigma*sigma))
        w = w / (np.sum(w, axis=0, keepdims=True) + 1e-8)
        merged_lin = np.sum(stack * w[...,None], axis=0)
                      
        a = 0.055
        low = merged_lin <= 0.0031308
        out = np.empty_like(merged_lin)
        out[low]  = 12.92 * merged_lin[low]
        out[~low] = (1 + a) * (merged_lin[~low] ** (1/2.4)) - a
        out_u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return out_u8

def postprocess_night_city(img_u8, vibrance=0.10, sharp_amount=0.6, sharp_radius=2):
    """
    Gentle finishing: a bit of vibrance (HSV-based) and local contrast (unsharp mask).
    Tuned to avoid neon look while restoring midtone colorfulness.
    """
    import colorsys
    img = img_u8.astype(np.float32) / 255.0
    h, w, _ = img.shape
    out = np.empty_like(img)
                                                                            
    for y in range(h):
        row = img[y]
        for x in range(w):
            r,g,b = row[x]
            H,S,V = colorsys.rgb_to_hsv(r,g,b)
            S2 = np.clip(S * (1.0 + vibrance * (1.0 - V)), 0.0, 1.0)
            r2,g2,b2 = colorsys.hsv_to_rgb(H,S2,V)
            out[y,x] = (r2,g2,b2)
    try:
        import cv2
                              
        blur = cv2.GaussianBlur((out*255).astype(np.uint8), (0,0), sharp_radius)
        sharp = cv2.addWeighted((out*255).astype(np.uint8), 1+sharp_amount, blur, -sharp_amount, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)
    except Exception:
                                                                    
        k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        from scipy.signal import convolve2d                                         
        def conv2(ch):
            return convolve2d(ch, k, mode='same', boundary='symm')
        r = conv2((out[...,0]*255.0)); g = conv2((out[...,1]*255.0)); b = conv2((out[...,2]*255.0))
        merged = np.stack([r,g,b], axis=2)
        return np.clip(merged, 0, 255).astype(np.uint8)

def srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    """Convert 8-bit or float sRGB [0..255 or 0..1] to linear RGB in [0..1]."""
    x = arr.astype(np.float32)
    if x.max() > 1.0001:
        x = x / 255.0
                       
    a = 0.055
    low = x <= 0.04045
    out = np.empty_like(x, dtype=np.float32)
    out[low]  = x[low] / 12.92
    out[~low] = ((x[~low] + a) / (1 + a)) ** 2.4
    return np.clip(out, 0.0, 1.0)

def linear_to_srgb(arr_lin: np.ndarray) -> np.ndarray:
    """Convert linear RGB [0..1] to 8-bit sRGB."""
    x = np.clip(arr_lin, 0.0, 1.0).astype(np.float32)
    a = 0.055
    low = x <= 0.0031308
    out = np.empty_like(x, dtype=np.float32)
    out[low]  = 12.92 * x[low]
    out[~low] = (1 + a) * (x[~low] ** (1/2.4)) - a
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

def tonemap_luminance_preserving_lin(linear_rgb: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Tonemap in linear space preserving chroma:
    1) compute Y (luminance)
    2) apply tone curve to Y only (Reinhard: Y_t = Y/(1+Y))
    3) rescale RGB by (Y_t / (Y+eps))^p to keep colorfulness; p ~ 0.9..1.1 tunes saturation.
    Returns linear RGB in [0..1].
    """
                                                    
    R, G, B = linear_rgb[..., 0], linear_rgb[..., 1], linear_rgb[..., 2]
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

                     
    Y_t = Y / (1.0 + Y)

                                                                       
    p = 1.0                                                   
    scale = (Y_t / (Y + eps)) ** p
    out = linear_rgb * scale[..., None]

                                               
    return np.clip(out, 0.0, 1.0)

def add_vibrance_srgb(img_u8: np.ndarray, gain: float = 0.1) -> np.ndarray:
    """
    Low-impact saturation boost in sRGB:
    - convert to HSV
    - S' = S * (1 + gain * (1 - V))  # less boost on highlights, more on midtones
    """
    import colorsys
    img = img_u8.astype(np.float32) / 255.0
    h, w, _ = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            r, g, b = img[y, x]
                                           
            H, S, V = colorsys.rgb_to_hsv(r, g, b)
            S2 = np.clip(S * (1.0 + gain * (1.0 - V)), 0.0, 1.0)
            r2, g2, b2 = colorsys.hsv_to_rgb(H, S2, V)
            out[y, x] = (r2, g2, b2)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

def _try_import_user_pkg():
    """Attempt to import the user's refactored package if present.

    English:
    - Also auto-detect the package by using the folder where this script lives.
    - Insert the parent directory to sys.path and try importing by the folder name.
    """
    import sys, os
    from pathlib import Path
    global PKG_TRY_DONE, ResponseFunction, HDRCoreProcessor, TonemapEnum
    if PKG_TRY_DONE:
        return
    PKG_TRY_DONE = True

                                     
    pkg_path = os.environ.get("HDR_PKG_PATH")
    if pkg_path and os.path.isdir(pkg_path) and pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)

                                                                                      
    here = Path(__file__).resolve().parent                                             
    parent = here.parent                                                  
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

    dynamic_name = here.name                                                         

    candidates = [dynamic_name, "refactored_hdr_package", "refactored_pkg", "hdr_refactored", "hdr"]

    for name in candidates:
        try:
            mod = __import__(name, fromlist=["*"])
                              
            RF = None
            if hasattr(mod, "ResponseFunction"):
                RF = getattr(mod, "ResponseFunction")
            else:
                try:
                    resp = __import__(f"{name}.response_function", fromlist=["ResponseFunction"])
                    RF = getattr(resp, "ResponseFunction", None)
                except Exception:
                    pass
                              
            HCP = None
            try:
                hcp = __import__(f"{name}.hdr_core_processor", fromlist=["HDRCoreProcessor"])
                HCP = getattr(hcp, "HDRCoreProcessor", None)
            except Exception:
                pass
                                     
            TM = None
            try:
                en = __import__(f"{name}.utils.enums", fromlist=["TonemappingAlgorithm"])
                TM = getattr(en, "TonemappingAlgorithm", None)
            except Exception:
                pass

            if RF is not None:
                ResponseFunction = RF
                HDRCoreProcessor = HCP
                TonemapEnum = TM
                print(f"[demo] Using user's package from '{name}'")
                return
        except Exception:
            continue

    print("[demo] User package not found; will use built-in minimal implementations.")

                                
                               
                                

class _LocalResponseFunction:
    """Minimal LS fit y = a*x + b with simple monotonicity guard (demo only)."""
    def __init__(self, parameter_a: float = 1.0, parameter_b: float = 0.0):
        self.parameter_a = float(parameter_a)
        self.parameter_b = float(parameter_b)

    @staticmethod
    def create_from_samples(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None, func_id: int = 0):
        x = x.astype(np.float64).ravel()
        y = y.astype(np.float64).ravel()
        if w is None:
            w = np.ones_like(x)
        else:
            w = w.astype(np.float64).ravel()
        sw = np.sum(w)
        swx = np.sum(w * x)
        swy = np.sum(w * y)
        swxx = np.sum(w * x * x)
        swxy = np.sum(w * x * y)
        denom = sw * swxx - swx * swx
        if abs(denom) < 1e-8:
                                  
            return _LocalResponseFunction(1.0, 0.0)
        a = (sw * swxy - swx * swy) / denom
        b = (swy - a * swx) / sw
        if a < 1e-6:
            a = 1.0
        if b < 0.0:
            b = 0.0
        return _LocalResponseFunction(a, b)


def _tonemap_reinhard(rgb: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Global Reinhard tonemapping per channel.
    Input: float32 array in [0..255].
    Output: uint8 in [0..255].
    """
    x = rgb.astype(np.float32) / 255.0
                                     
    y = x / (1.0 + x + eps)
    out = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return out


def _compute_mtb(gray: np.ndarray, min_diff: int = 4) -> Tuple[np.ndarray, int]:
    """Compute a simple Median Threshold Bitmap and return (mtb_mask, median)."""
    med = int(np.median(gray))
    low, high = max(0, med - min_diff), min(255, med + min_diff)
    mtb = np.where((gray < low) | (gray > high), (gray > med).astype(np.uint8) * 255, 0)
    return mtb.astype(np.uint8), med


def _align_images_mtb(images: List[Image.Image]) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """
    Try OpenCV's AlignMTB. If unavailable, fallback to identity alignment.
    Returns aligned PIL images and list of MTB masks for visualization.
    """
    mtb_masks = []
    if HAS_CV2:
        try:
            cv_imgs = [cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR) for im in images]
            align = cv2.createAlignMTB()
            align.process(cv_imgs, cv_imgs)
            aligned_pil = [Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)) for a in cv_imgs]
                                   
            for im in aligned_pil:
                g = np.array(im.convert("L"))
                mtb, _ = _compute_mtb(g)
                mtb_masks.append(mtb)
            return aligned_pil, mtb_masks
        except Exception as e:
            print(f"[demo] OpenCV AlignMTB failed: {e}; falling back to identity alignment.")
                                                                    
    aligned = images
    for im in images:
        g = np.array(im.convert("L"))
        mtb, _ = _compute_mtb(g)
        mtb_masks.append(mtb)
    return aligned, mtb_masks


def _auto_resize_to_common(images: List[Image.Image]) -> List[Image.Image]:
    """Resize all images to the minimum common size to avoid size mismatch errors."""
    widths, heights = zip(*[im.size for im in images])
    w, h = min(widths), min(heights)
    out = [im.resize((w, h), Image.BICUBIC) if im.size != (w, h) else im for im in images]
    return out


def _estimate_response_functions(images: List[Image.Image], base_idx: int = 0):
    """
    Estimate response functions for each image relative to the base image.
    Returns a list of (a, b) for y = a*x + b. Uses user's ResponseFunction if available.
    """
    _try_import_user_pkg()
    RF = ResponseFunction if ResponseFunction is not None else _LocalResponseFunction

    base = np.array(images[base_idx], dtype=np.float32)
    base_gray = np.mean(base, axis=2)

    ab = []
    for i, im in enumerate(images):
        if i == base_idx:
            ab.append((1.0, 0.0))
            continue
        arr = np.array(im, dtype=np.float32)
        arr_gray = np.mean(arr, axis=2)

                                                               
        rng = np.random.default_rng(1234 + i)
        h, w = base_gray.shape
        ys = rng.integers(0, h, size=30000)
        xs = rng.integers(0, w, size=30000)
        x_vals = base_gray[ys, xs]
        y_vals = arr_gray[ys, xs]
        wts = np.ones_like(x_vals, dtype=np.float32)

        rf = RF.create_from_samples(x_vals, y_vals, wts, func_id=0)
        a = float(getattr(rf, "parameter_a", getattr(rf, "parameter_A", 1.0)))
        b = float(getattr(rf, "parameter_b", getattr(rf, "parameter_B", 0.0)))
        ab.append((a, b))
    return ab

def _merge_and_tonemap(images: List[Image.Image],
                       ab: List[Tuple[float, float]],
                       tonemap: str = "REINHARD",
                       use_weights: bool = True,
                       vibrance_gain: float = 0.15) -> np.ndarray:
    """
    Merge in *linear* space, tonemap luminance-preserving, then (optionally) a mild vibrance.
    - 'ab' are still used, but applied in *sRGB* space before linearization (as in your demo).
      If you have a response function that maps sensor->linear directly, apply it *before* srgb_to_linear.
    """
                                                                           
    arrs_lin = []
    for i, im in enumerate(images):
        arr = np.asarray(im, dtype=np.float32)            
        a, b = ab[i]
        arr = a * arr + b
        lin = srgb_to_linear(arr)             
        arrs_lin.append(lin)

                                       
    stack = np.stack(arrs_lin, axis=0)              
    Y = 0.2126 * stack[..., 0] + 0.7152 * stack[..., 1] + 0.0722 * stack[..., 2]
    if use_weights:
                                              
        sigma = 0.25
        w = np.exp(-((Y - 0.5) ** 2) / (2 * sigma * sigma))           
        w = w / (np.sum(w, axis=0, keepdims=True) + 1e-8)
        merged_lin = np.sum(stack * w[..., None], axis=0)
    else:
        merged_lin = np.mean(stack, axis=0)

                                                        
    tm_lin = tonemap_luminance_preserving_lin(merged_lin)

                     
    out_u8 = linear_to_srgb(tm_lin)

                                
    if vibrance_gain > 1e-6:
        out_u8 = add_vibrance_srgb(out_u8, gain=vibrance_gain)

    return out_u8


def _save_histogram(image: Image.Image, title: str, path: str):
    """Plot a single histogram of grayscale intensities and save it."""
    g = np.array(image.convert("L")).ravel()
    plt.figure()
    plt.hist(g, bins=256, range=(0, 255))
    plt.title(title)
    plt.xlabel("Gray value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _save_response_plot(x: np.ndarray, y: np.ndarray, a: float, b: float, title: str, path: str):
    """Plot scatter (downsampled) + fit line for response function and save."""
    plt.figure()
                                  
    idx = np.linspace(0, len(x) - 1, num=min(5000, len(x))).astype(int)
    plt.scatter(x[idx], y[idx], s=2)
          
    x_line = np.linspace(0, 255, 256)
    y_line = a * x_line + b
    plt.plot(x_line, y_line)
    plt.title(title)
    plt.xlabel("Base gray (X)")
    plt.ylabel("Other gray (Y)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _make_collage(images: List[Image.Image], cols: int, out_path: str, cell_size: Optional[Tuple[int,int]] = None):
    """Make a simple grid collage from PIL images and save it."""
    if not images:
        return
    if cell_size is None:
        cell_w = min(im.width for im in images)
        cell_h = min(im.height for im in images)
    else:
        cell_w, cell_h = cell_size
    ims = [im.resize((cell_w, cell_h), Image.BICUBIC) for im in images]
    rows = (len(ims) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        canvas.paste(im, (c * cell_w, r * cell_h))
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="4 input images (paths)")
    ap.add_argument("--out", default="hdr_out", help="Output directory")
    ap.add_argument("--tonemap", default="REINHARD", help="Tonemap: REINHARD|CLAMP")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

                 
    paths = args.images
    if len(paths) != 4:
        print("[demo] Please provide exactly 4 images for this demo.")
        sys.exit(2)

    imgs = [Image.open(p).convert("RGB") for p in paths]
    imgs = _auto_resize_to_common(imgs)

                            
    _make_collage(imgs, cols=4, out_path=os.path.join(args.out, "00_inputs_collage.jpg"))

                                    
    aligned, mtb_masks = _align_images_mtb(imgs)
    _make_collage(aligned, cols=4, out_path=os.path.join(args.out, "01_aligned_collage.jpg"))
    mtb_vis = [Image.fromarray(m) for m in mtb_masks]
    _make_collage(mtb_vis, cols=4, out_path=os.path.join(args.out, "02_mtb_masks_collage.jpg"))

                                   
    for i, im in enumerate(aligned):
        _save_histogram(im, f"Histogram image {i}", os.path.join(args.out, f"10_hist_{i}.png"))

                                                                                             
    medians = [int(np.median(np.array(im.convert("L")))) for im in aligned]
    base_idx = int(np.argsort([abs(m - 128) for m in medians])[0])
    ab = _estimate_response_functions(aligned, base_idx=base_idx)

                                      
    base_gray = np.mean(np.array(aligned[base_idx], dtype=np.float32), axis=2)
    for i, im in enumerate(aligned):
        if i == base_idx:
            continue
        g = np.mean(np.array(im, dtype=np.float32), axis=2)
        rng = np.random.default_rng(1234 + i)
        h, w = g.shape
        ys = rng.integers(0, h, size=30000); xs = rng.integers(0, w, size=30000)
        x_vals = base_gray[ys, xs]; y_vals = g[ys, xs]
        a, b = ab[i]
        _save_response_plot(x_vals, y_vals, a, b,
                            title=f"Response fit (base={base_idx} vs {i})",
                            path=os.path.join(args.out, f"20_response_fit_{base_idx}_vs_{i}.png"))

                     

    hdr_ldr = exposure_fusion_mertens(aligned, contrast=1.0, saturation=0.5, well_exposed=1.5)
    hdr_ldr = postprocess_night_city(hdr_ldr, vibrance=0.10, sharp_amount=0.55, sharp_radius=1.6)
    out_img = Image.fromarray(hdr_ldr)

    out_img.save(os.path.join(args.out, "99_hdr_tonemapped.jpg"))

    print("[demo] Outputs saved to:", os.path.abspath(args.out))
    print(" - 00_inputs_collage.jpg   (originals)")
    print(" - 01_aligned_collage.jpg  (aligned)")
    print(" - 02_mtb_masks_collage.jpg")
    print(" - 10_hist_*.png           (per-image histograms)")
    print(" - 20_response_fit_*.png   (response function fits)")
    print(" - 99_hdr_tonemapped.jpg   (final LDR HDR)")

if __name__ == "__main__":
    main()
