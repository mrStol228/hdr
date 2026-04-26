# HDR Pipeline

A Python library for **multi-exposure HDR imaging** — exposure fusion, tone mapping, multi-frame denoising, and image alignment — originally designed for deployment on NVIDIA Jetson edge hardware.

---

## Features

- **Exposure Fusion (Mertens)** — merge bracketed exposures into a natural-looking LDR image
- **N-frame HDR merge** — full linear-domain merge with luminance-preserving tone mapping (Reinhard, ACES, Exponential, Clamp)
- **Temporal Wiener denoising** — multi-frame noise reduction with de-ghosting (ported from a Java camera pipeline)
- **Response Function fitting** — per-image linear least-squares exposure calibration (`y = A·x + B`)
- **MTB image alignment** — hierarchical Median Threshold Bitmap alignment (works without OpenCV)
- **Single-image DRO** — Dynamic Range Optimization with adaptive gain/gamma and CLAHE
- **Multi-image averaging** — robust weighted average with outlier rejection
- **OpenCV acceleration** — `AlignMTB`, `MergeMertens`, and `fastNlMeansDenoising` used automatically when OpenCV is installed; graceful pure-NumPy fallback otherwise

---

## Architecture

```
hdr_pipeline/
├── hdr_core_processor.py      ← Main entry point: HDRCoreProcessor
│     ├── exposure_fusion_mertens()   Mertens LDR fusion
│     ├── apply_hdr_n_function()      N-frame HDR merge + tonemap
│     └── temporal_wiener_denoise()   Multi-frame NR
│
├── response_function.py       ← Exposure response fitting (y = A·x + B)
├── single_image_processor.py  ← DRO + CLAHE for a single frame
├── image_averaging_processor.py ← Wiener-filtered frame averaging (NR mode)
├── multi_image_processor.py   ← 8-frame burst averaging with MTB alignment
├── image_functions.py         ← Low-level pixel kernels (tonemapping, histogram, MTB)
└── utils/
      ├── enums.py             ← TonemappingAlgorithm, HistogramType, HDRAlgorithm
      ├── data_classes.py      ← HistogramInfo, BrightenFactors, AvgData, …
      └── exceptions.py        ← HDRProcessorException
```

---

## Installation

```bash
git clone https://github.com/<your-username>/hdr-pipeline.git
cd hdr-pipeline
pip install -r requirements.txt

# Optional: GPU-accelerated paths via OpenCV
pip install opencv-python
```

---

## Quick Start

### Exposure Fusion (Mertens)

```python
from PIL import Image
from hdr_pipeline import HDRCoreProcessor

images = [Image.open(p) for p in ["dark.jpg", "mid.jpg", "bright.jpg", "overexp.jpg"]]

core = HDRCoreProcessor()
fused = core.exposure_fusion_mertens(
    images,
    align=True,           # MTB alignment (OpenCV if available)
    pre_denoise=True,     # shadow denoising before merge
    post_chroma_denoise=True,
)
Image.fromarray(fused).save("fused.jpg")
```

### HDR Merge with Tone Mapping

```python
from hdr_pipeline import HDRCoreProcessor, ResponseFunction

core = HDRCoreProcessor(enable_temporal_nr=True)

# Fit response functions (exposure mapping between frames)
# You can use demo_validate.py to compute these automatically
parameters_a = [0.7, 1.0, 1.5, 2.0]  # one per image
parameters_b = [0.0, 0.0, 0.0, 0.0]

result = core.apply_hdr_n_function(
    bitmaps=images,
    base_index=1,
    offsets_x=[0, 0, 0, 0],
    offsets_y=[0, 0, 0, 0],
    parameters_a=parameters_a,
    parameters_b=parameters_b,
    tonemapping_algorithm="REINHARD",  # REINHARD | ACES | EXPONENTIAL | CLAMP
)
result.save("hdr_out.jpg")
```

### Response Function Fitting

```python
from hdr_pipeline import ResponseFunction

# x_samples: pixel values from base image
# y_samples: corresponding pixels from another exposure
rf = ResponseFunction.create_from_samples(x_samples, y_samples, weights)
print(f"A={rf.parameter_a:.4f}  B={rf.parameter_b:.4f}")

mapped_value = rf.apply(128.0)
```

---

## Demo Scripts

### `examples/demo_hdr.py` — Full HDR pipeline demo

Loads 4 images, aligns them, fits response functions, runs Mertens fusion and saves a full visual report.

```bash
python examples/demo_hdr.py \
    --images dark.jpg mid_dark.jpg mid_bright.jpg bright.jpg \
    --out output_dir \
    --tonemap REINHARD
```

**Outputs:**
- `00_inputs_collage.jpg` — all inputs side by side
- `01_aligned_collage.jpg` — after MTB alignment
- `02_mtb_masks_collage.jpg` — MTB visualisation
- `10_hist_*.png` — per-image histograms
- `20_response_fit_*.png` — response function scatter plots
- `99_hdr_tonemapped.jpg` — final result

---

### `examples/demo_validate.py` — Validation runner

More detailed pipeline with separate Mertens and HDR-merge paths, response-function fitting, and an HTML report.

```bash
python examples/demo_validate.py \
    --images img0.jpg img1.jpg img2.jpg img3.jpg \
    --out validate_out \
    --pipeline both \
    --hdr-tonemap REINHARD \
    --pre-denoise
```

**Outputs:** HTML report at `validate_out/index.html` with all intermediate images.

---

## Tonemapping Algorithms

| Name | Description |
|------|-------------|
| `REINHARD` | `Y_t = Y / (1 + Y)` — smooth, natural (default) |
| `ACES` | Filmic S-curve — rich contrast |
| `EXPONENTIAL` | `Y_t = 1 - exp(-Y)` — soft highlights |
| `CLAMP` | Hard clip — fastest, useful for debugging |

---

## OpenCV Acceleration

| Feature | Without OpenCV | With OpenCV |
|---------|---------------|-------------|
| Image alignment | Pure-Python MTB | `cv2.createAlignMTB()` |
| Exposure fusion | Weighted linear blend | `cv2.createMergeMertens()` |
| Pre-denoise | 3×3 median in shadows | `fastNlMeansDenoisingColored` |
| Chroma denoise | Box-blur in YCbCr | `fastNlMeansDenoising` per channel |

Install with: `pip install opencv-python`

---

## Project Background

This library was developed as part of a **real-time multi-camera computer vision pipeline** running on **NVIDIA Jetson** hardware. The core algorithms (Wiener averaging, MTB alignment, piecewise gain/gamma, CLAHE) were ported from a production Java camera stack to Python, then extended with additional features for edge deployment.

Key use cases:
- HDR video pipelines (DeepStream / GStreamer integration)
- Difficult lighting conditions (high-contrast scenes, backlit subjects)
- Low-light burst photography
- Smart city / surveillance camera image enhancement

---

## Examples and Test Images

To run the demos, you'll need to provide bracketed exposure images. Place your test images in the `examples/images/` directory.

### Recommended Test Images

For best results, upload **3-5 images** of the same scene with different exposures (bracketed shots):

1. **Under-exposed** (dark, EV -2 to -1): Captures highlights without clipping
2. **Mid-exposed** (normal, EV 0): Reference exposure
3. **Over-exposed** (bright, EV +1 to +2): Captures shadows without underexposure

Example filenames: `scene_dark.jpg`, `scene_mid.jpg`, `scene_bright.jpg`

For multi-frame denoising, use **8-10 identical exposures** (burst mode) of a static scene.

### Running Demos

```bash
# Exposure fusion demo
python examples/demo_hdr.py --images examples/images/scene_dark.jpg examples/images/scene_mid.jpg examples/images/scene_bright.jpg --out output/

# Response function validation
python examples/demo_validate.py --images examples/images/*.jpg
```

---

## License

MIT
