#!/usr/bin/env python3
"""
simple_hdr_demo.py
------------------
Simple demo for HDR processing with 3 bracketed images.

Usage:
    python examples/simple_hdr_demo.py
    (Assumes images are in examples/images/ as scene_dark.jpg, scene_mid.jpg, scene_bright.jpg)
"""

import os
import sys
from PIL import Image
from hdr_pipeline import HDRCoreProcessor

def main():
    # Expected image paths
    image_dir = "examples/images"
    image_names = ["scene_dark.jpg", "scene_mid.jpg", "scene_bright.jpg"]
    image_paths = [os.path.join(image_dir, name) for name in image_names]

    # Check if images exist
    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        print(f"Missing images: {missing}")
        print("Please place your bracketed exposure images in examples/images/")
        print("Recommended: scene_dark.jpg (under-exposed), scene_mid.jpg (normal), scene_bright.jpg (over-exposed)")
        sys.exit(1)

    # Load images
    images = [Image.open(p).convert("RGB") for p in image_paths]
    print(f"Loaded {len(images)} images")

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize HDR processor
    core = HDRCoreProcessor()

    # Exposure fusion (Mertens)
    print("Performing exposure fusion...")
    fused = core.exposure_fusion_mertens(
        images,
        align=True,
        pre_denoise=True,
        post_chroma_denoise=True
    )

    # Save result
    output_path = os.path.join(output_dir, "fused_result.jpg")
    Image.fromarray(fused).save(output_path)
    print(f"Saved fused image to {output_path}")

    print("Demo completed!")

if __name__ == "__main__":
    main()