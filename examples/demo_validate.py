from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import logging
from typing import List, Tuple

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = logging.getLogger("demo")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def _try_import_user_pkg():
    """
    Insert candidate paths and try importing the user's package.
    We prioritize the folder where this script lives.
    """
    here = Path(__file__).resolve().parent
    parent = here.parent
    candidates = [here.name, "refactored_hdr_package", "refactored_pkg", "hdr_refactored", "hdr"]
    pkg_path = os.environ.get("HDR_PKG_PATH")
    if pkg_path and pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

    for name in candidates:
        try:
            mod = __import__(name, fromlist=["*"])
                              
            core = __import__(f"{name}.hdr_core_processor", fromlist=["HDRCoreProcessor"])
            HDRCoreProcessor = getattr(core, "HDRCoreProcessor")
            resp = __import__(f"{name}.response_function", fromlist=["ResponseFunction"])
            ResponseFunction = getattr(resp, "ResponseFunction")
            try:
                enums = __import__(f"{name}.utils.enums", fromlist=["TonemappingAlgorithm", "HDRAlgorithm"])
                TonemappingAlgorithm = getattr(enums, "TonemappingAlgorithm", None)
                HDRAlgorithm = getattr(enums, "HDRAlgorithm", None)
            except Exception:
                TonemappingAlgorithm = None
                HDRAlgorithm = None
            LOG.info("Using user's package: %s", name)
            return name, HDRCoreProcessor, ResponseFunction, TonemappingAlgorithm, HDRAlgorithm
        except Exception as e:
            continue
    raise RuntimeError("User package not found. Set HDR_PKG_PATH or place this script next to the package folder.")

PKG_NAME, HDRCoreProcessor, ResponseFunction, TonemapEnum, HDRAlgorithm = _try_import_user_pkg()

                                                                                        
               
                                                                                        

def _ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _load_images(paths: List[Path]) -> List[Image.Image]:
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        ims.append(im)
                                                   
    Hs = [im.height for im in ims]
    Ws = [im.width for im in ims]
    Hmin, Wmin = min(Hs), min(Ws)
    ims = [im.resize((Wmin, Hmin), Image.BILINEAR) for im in ims]
    return ims

def _srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)
    if x.max() > 1.0001:
        x = x / 255.0
    a = 0.055
    low = x <= 0.04045
    out = np.empty_like(x, dtype=np.float32)
    out[low] = x[low] / 12.92
    out[~low] = ((x[~low] + a) / (1 + a)) ** 2.4
    return np.clip(out, 0.0, 1.0)

def _collage_h(images: List[Image.Image]) -> Image.Image:
    """Horizontal collage."""
    if not images: raise ValueError("no images")
    w, h = images[0].size
    out = Image.new("RGB", (w * len(images), h))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += w
    return out

def _histogram_plot(img: Image.Image, path: Path, title: str):
    arr = np.asarray(img, dtype=np.uint8)
    Y = (0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]).astype(np.uint8)
    plt.figure()
    plt.hist(Y.ravel(), bins=256, range=(0,255))
    plt.title(title)
    plt.xlabel("Luma (Rec.709)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _estimate_noise_u8(img: Image.Image) -> float:
    """Rough noise estimate (MAD-based) on darkest 15% of luma."""
    arr = np.asarray(img, dtype=np.uint8)
    Y = 0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]
    thr = np.percentile(Y, 15.0)
    dark = arr[Y <= thr]
    if dark.size == 0:
        return 0.0
    diff = dark.astype(np.float32) - np.median(dark, axis=0)
    mad = np.median(np.abs(diff))
    return float(1.4826 * mad)

def _fit_response_pair(base: Image.Image, other: Image.Image, n_samples: int = 30000,
                       exclude=(5,95)) -> Tuple[float, float]:
    """
    Fit ResponseFunction between 'base' and 'other' via random sampling.
    Compatible with current ResponseFunction API: requires weights and no extra kwargs.
    NOTE: We fit in sRGB (luma) domain to stay consistent with current HDR core mapping (a*x+b on sRGB).
    """
    a = np.asarray(base, dtype=np.uint8)
    b = np.asarray(other, dtype=np.uint8)
    H, W, _ = a.shape
    rng = np.random.default_rng(42)
    ys = rng.integers(0, H, size=n_samples)
    xs = rng.integers(0, W, size=n_samples)
                           
    X = (0.2126*a[ys, xs, 0] + 0.7152*a[ys, xs, 1] + 0.0722*a[ys, xs, 2]).astype(np.float32)
    Y = (0.2126*b[ys, xs, 0] + 0.7152*b[ys, xs, 1] + 0.0722*b[ys, xs, 2]).astype(np.float32)
                      
    xlo, xhi = np.percentile(X, exclude)
    ylo, yhi = np.percentile(Y, exclude)
    m = (X >= xlo) & (X <= xhi) & (Y >= ylo) & (Y <= yhi)
    X, Y = X[m], Y[m]
    Wts = np.ones_like(X, dtype=np.float32)                                        
    rf = ResponseFunction.create_from_samples(X.tolist(), Y.tolist(), Wts.tolist(), 0)
                                                   
    return float(rf.parameter_a), float(rf.parameter_b)

def _save_response_plot(base: Image.Image, other: Image.Image, A: float, B: float, path: Path, title: str):
    a = np.asarray(base, dtype=np.uint8)
    b = np.asarray(other, dtype=np.uint8)
    H, W, _ = a.shape
    rng = np.random.default_rng(123)
    ys = rng.integers(0, H, size=8000)
    xs = rng.integers(0, W, size=8000)
    x = (0.2126*a[ys, xs, 0] + 0.7152*a[ys, xs, 1] + 0.0722*a[ys, xs, 2]).astype(np.float32)
    y = (0.2126*b[ys, xs, 0] + 0.7152*b[ys, xs, 1] + 0.0722*b[ys, xs, 2]).astype(np.float32)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(x, y, s=2, alpha=0.3)
    xs = np.linspace(0, 255, 256, dtype=np.float32)
    plt.plot(xs, A*xs + B)
    plt.title(title)
    plt.xlabel("Base luma")
    plt.ylabel("Other luma")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

                                                                                        
      
                                                                                        

def main():
    ap = argparse.ArgumentParser(description="HDR demo/validation runner (v2)")
    ap.add_argument("--images", nargs=4, required=True, help="Four image paths (different ISO/exposures)")
    ap.add_argument("--out", default="hdr_validate_out", help="Output directory")
    ap.add_argument("--pipeline", choices=["mertens", "hdr", "both"], default="both",
                    help="Which pipeline to run")
    ap.add_argument("--align", action="store_true", help="Try AlignMTB where applicable (OpenCV)")
                    
    ap.add_argument("--m-contrast", type=float, default=1.0)
    ap.add_argument("--m-saturation", type=float, default=0.5)
    ap.add_argument("--m-well", type=float, default=1.5)
    ap.add_argument("--pre-denoise", action="store_true", help="Pre-denoise frames for Mertens/HDR where applicable")
    ap.add_argument("--pre-strength", default="auto", help='"auto" or 0..1')
    ap.add_argument("--post-chroma-denoise", action="store_true", help="Post chroma-only denoise after fusion")
    ap.add_argument("--post-strength", type=float, default=0.15)
                
    ap.add_argument("--hdr-weights", action="store_true", help="Use well-exposedness weights for HDR merge")
    ap.add_argument("--hdr-tonemap", default="REINHARD", help="REINHARD|CLAMP|EXPONENTIAL|ACES")
    ap.add_argument("--hdr-linear-scale", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    _ensure_outdir(out_dir)

                               
    img_paths = [Path(p) for p in args.images]
    images = _load_images(img_paths)

                    
    _collage_h(images).save(out_dir / "00_inputs_collage.jpg", quality=95)
    for i, im in enumerate(images):
        _histogram_plot(im, out_dir / f"10_hist_input_{i}.png", f"Input {i} histogram")

    core = HDRCoreProcessor(enable_temporal_nr=True)

                                            
    if args.pipeline in ("mertens", "both"):
        fused = core.exposure_fusion_mertens(
            images,
            align=args.align,
            contrast=args.m_contrast,
            saturation=args.m_saturation,
            well_exposed=args.m_well,
            pre_denoise=args.pre_denoise,
            pre_strength=args.pre_strength,
            post_chroma_denoise=args.post_chroma_deniose if hasattr(args, 'post_chroma_deniose') else args.post_chroma_denoise,
            post_strength=args.post_strength
        )
        Image.fromarray(fused).save(out_dir / "50_mertens_fused.jpg", quality=95)
        _histogram_plot(Image.fromarray(fused), out_dir / "51_hist_mertens.png", "Mertens Histogram")
        LOG.info("Mertens fusion saved: %s", out_dir / "50_mertens_fused.jpg")

                                        
    if args.pipeline in ("hdr", "both"):
        base_idx = 1                                             
                                             
        A_vals, B_vals = [], []
        for i, im in enumerate(images):
            if i == base_idx:
                A_vals.append(1.0); B_vals.append(0.0)
            else:
                A, B = _fit_response_pair(images[base_idx], im, n_samples=30000, exclude=(5,95))
                A_vals.append(float(A)); B_vals.append(float(B))
                _save_response_plot(images[base_idx], im, A, B, out_dir / f"20_response_fit_{i}.png",
                                    f"Response fit base({base_idx}) -> img({i})")

        offsets_x = [0, 0, 0, 0]
        offsets_y = [0, 0, 0, 0]

        out_img = core.apply_hdr_n_function(
            bitmaps=images,
            base_index=base_idx,
            offsets_x=offsets_x,
            offsets_y=offsets_y,
            parameters_a=A_vals,
            parameters_b=B_vals,
            tonemapping_algorithm=args.hdr_tonemap,
            linear_scale=args.hdr_linear_scale,
            use_weights=args.hdr_weights
        )
        out_img.save(out_dir / "90_hdr_tonemapped.jpg", quality=95)
        _histogram_plot(out_img, out_dir / "91_hist_hdr.png", "HDR Histogram")
        LOG.info("HDR tonemapped saved: %s", out_dir / "90_hdr_tonemapped.jpg")

                                              
    html = []
    html.append("<html><head><meta charset='utf-8'><title>HDR Demo Report</title></head><body>")
    html.append("<h1>HDR Demo/Validation Report</h1>")
    html.append("<h2>Inputs</h2>")
    html.append("<img src='00_inputs_collage.jpg' style='max-width:100%;'><br>")
    for i in range(4):
        html.append(f"<img src='10_hist_input_{i}.png' width='380'>")
    if (out_dir / "50_mertens_fused.jpg").exists():
        html.append("<h2>Mertens Fusion</h2>")
        html.append("<img src='50_mertens_fused.jpg' style='max-width:100%;'><br>")
        html.append("<img src='51_hist_mertens.png' width='420'>")
    if (out_dir / "90_hdr_tonemapped.jpg").exists():
        html.append("<h2>HDR Merge</h2>")
        for i in range(4):
            if (out_dir / f"20_response_fit_{i}.png").exists():
                html.append(f"<img src='20_response_fit_{i}.png' width='420'>")
        html.append("<br>")
        html.append("<img src='90_hdr_tonemapped.jpg' style='max-width:100%;'><br>")
        html.append("<img src='91_hist_hdr.png' width='420'>")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    LOG.info("HTML report: %s", out_dir / "index.html")


if __name__ == "__main__":
    main()
