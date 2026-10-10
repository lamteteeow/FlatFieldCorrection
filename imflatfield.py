"""Flat-field correction for 16-bit grayscale TIFF images.

The illumination field is estimated *per frame* as the frame blurred with a
Gaussian of standard deviation ``sigma`` (OpenCV), normalised to unit mean, and
the frame is divided by it: ``img / field * mean(field)``.  Only structure
coarser than ``sigma`` is flattened; no dark reference is used (see the TODO in
README.md), so the result is a relative normalisation, not quantitative
transmission.

Interactive use (file dialog + sigma prompt):

    uv run imflatfield.py

Headless batch use:

    uv run imflatfield.py --input FRAMES_DIR --sigma 20 --out OUT_DIR

``--input`` takes TIFF files and/or directories (directories are scanned
non-recursively for ``*.tif``/``*.tiff``).  Outputs keep the input basenames,
are written as uncompressed 16-bit grayscale TIFF, and existing outputs are
skipped unless ``--overwrite`` is given.  The exit status is non-zero if any
frame fails to read, convert or write.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2  # PIL builtin gaussian filter does not support float value
import numpy as np

TIFF_SUFFIXES = (".tif", ".tiff")
UINT16_MAX = int(np.iinfo(np.uint16).max)
# Prevent division by zero in the flat-field estimate.
EPSILON = 1e-12
# libtiff compression code 1 == COMPRESSION_NONE: keeps outputs plain, matching
# the per-frame TIFFs the rest of the pipeline works with, and readable by
# tifffile without its optional imagecodecs dependency.
TIFF_UNCOMPRESSED = 1


def estimate_flat_field(img_array, sigma=50):
    """Estimate flat field using low-frequency components"""
    # Create smoothed version for flat field estimation
    flat_estimate = cv2.GaussianBlur(img_array, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Normalize to maintain average intensity
    flat_estimate = flat_estimate / np.mean(flat_estimate)
    return flat_estimate


def estimate_dark_field():
    pass


def correct_frame(img_array, sigma, clip=False):
    """Divide a float64 frame by its self-estimated illumination field.

    Returns ``(uint16 frame, pixels above 65535)``.  Without ``clip`` those
    pixels wrap around at the cast, exactly like the original script; with it
    they saturate instead.
    """
    flat_field = estimate_flat_field(img_array, sigma)

    # Apply flat field correction with epsilon to prevent division by zero
    corrected = (img_array / (flat_field + EPSILON)) * np.mean(flat_field)

    overflow = int(np.count_nonzero(corrected > UINT16_MAX))
    if clip:
        corrected = np.clip(corrected, 0, UINT16_MAX)
    return corrected.astype(np.uint16), overflow


def _read_frame(file_path):
    """Read one frame as a 2D uint16 array, failing loudly on anything else."""
    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not open {file_path}")
    if img.ndim != 2 or img.dtype != np.uint16:
        raise ValueError(
            f"expected 2D 16-bit grayscale, got {img.shape}/{img.dtype}"
        )
    return img


def process_image(file_path, sigma, output_dir):
    """Interactive single-file processing (never overwrites an existing output)"""
    try:
        img = _read_frame(file_path)
        final_array, _ = correct_frame(img.astype(np.float64), sigma)

        # Generate output path with increment if file exists
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, base_name)
        counter = 1

        while os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{name}_{counter}{ext}")
            counter += 1

        # Save result
        cv2.imwrite(output_path, final_array)
        print(f"Processed: {os.path.basename(output_path)}")

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")


def process_frame(file_path, sigma, output_dir, overwrite=False, clip=False):
    """Batch worker: correct one frame file into ``output_dir``.

    Returns ``(name, status, pixels above 65535)`` with status ``written`` or
    ``skipped``.  The result is renamed into place only once it is fully
    written, so an interrupted run never leaves a half-written TIFF behind.
    """
    cv2.setNumThreads(1)  # the pool already parallelises over frames
    source = Path(file_path)
    target = Path(output_dir) / source.name
    if target.exists() and not overwrite:
        return source.name, "skipped", 0

    final_array, overflow = correct_frame(
        _read_frame(source).astype(np.float64), sigma, clip
    )
    # Encode in memory and rename into place, so an interrupted run never
    # leaves a partially written frame behind (and the temp name, which does
    # not end in .tif, cannot be mistaken for one).
    ok, buffer = cv2.imencode(
        ".tif", final_array, [cv2.IMWRITE_TIFF_COMPRESSION, TIFF_UNCOMPRESSED]
    )
    if not ok:
        raise ValueError(f"could not encode {source.name} as TIFF")
    partial = target.with_name(f".{target.name}.part")
    partial.write_bytes(buffer.tobytes())
    os.replace(partial, target)
    return source.name, "written", overflow


def collect_inputs(inputs):
    """Expand files/directories into a frame list, rejecting output collisions."""
    files: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            found = sorted(
                child
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in TIFF_SUFFIXES
            )
            if not found:
                raise SystemExit(f"no TIFF files in directory: {path}")
            files.extend(found)
        elif path.is_file():
            if path.suffix.lower() not in TIFF_SUFFIXES:
                raise SystemExit(f"not a TIFF file: {path}")
            files.append(path)
        else:
            raise SystemExit(f"no such file or directory: {path}")

    duplicates = sorted(
        name for name, count in Counter(f.name for f in files).items() if count > 1
    )
    if duplicates:
        raise SystemExit(
            "these frame names occur more than once and would overwrite each "
            f"other in the output directory: {', '.join(duplicates[:5])}"
        )
    return files


def run_batch(args):
    """Correct every input frame, printing progress; returns the exit status."""
    files = collect_inputs(args.input)
    output_dir = args.out.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, len(files)))

    print(
        f"Flat-field correction (sigma={args.sigma}) of {len(files)} frames "
        f"-> {output_dir}  [{workers} worker(s)]"
    )
    done = written = skipped = overflow_frames = overflow_pixels = 0
    peak = 0
    started = time.monotonic()
    # One line per frame while a terminal is watching, otherwise every 5% so a
    # redirected log stays readable.
    step = 1 if sys.stdout.isatty() else max(1, len(files) // 20)
    worker = functools.partial(
        process_frame,
        sigma=args.sigma,
        output_dir=str(output_dir),
        overwrite=args.overwrite,
        clip=args.clip,
    )
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for name, status, overflow in pool.map(worker, (str(f) for f in files)):
                done += 1
                written += status == "written"
                skipped += status == "skipped"
                if overflow:
                    overflow_frames += 1
                    overflow_pixels += overflow
                    peak = max(peak, overflow)
                elapsed = time.monotonic() - started
                if done % step and done != len(files):
                    continue
                print(
                    f"\r  {done}/{len(files)} frames  "
                    f"({elapsed:.0f}s, {done / max(elapsed, 1e-9):.1f} fps)  "
                    f"written={written} skipped={skipped}",
                    end="",
                    flush=True,
                )
    except Exception as e:
        print(f"\nFailed: {e}")
        return 1
    print()

    if skipped:
        print(f"Skipped {skipped} frame(s) whose output already existed.")
    if overflow_frames:
        if args.clip:
            print(
                f"WARNING: {overflow_frames} frame(s) exceeded the uint16 range "
                f"({overflow_pixels} pixels in total, worst frame {peak}); those "
                "pixels were saturated at 65535, so they are NOT measurements."
            )
        else:
            print(
                f"WARNING: {overflow_frames} frame(s) exceeded the uint16 range "
                f"({overflow_pixels} pixels in total, worst frame {peak}); the "
                "uint16 cast wrapped them around. Re-run with --clip to saturate "
                "instead of wrapping."
            )
    print(f"Wrote {written} frame(s) to {output_dir}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="TIFF files and/or directories of frames to correct",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="output directory for the frames"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        required=True,
        help="Gaussian sigma in pixels: the spatial scale up to which "
        "illumination is estimated and removed (README suggests 10-30)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="parallel worker processes (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-process frames whose output already exists (default: skip them)",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="saturate corrected values above 65535 instead of letting the "
        "uint16 cast wrap them around",
    )
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        args = build_parser().parse_args(argv)
        if args.sigma <= 0:
            raise SystemExit("--sigma must be a positive number")
        return run_batch(args)

    import tkinter
    from tkinter import filedialog

    try:
        # Ask user to select TIFF files
        file_paths = filedialog.askopenfilenames(
            title="Select TIFF files", filetypes=[("TIFF files", "*.tif;*.tiff")]
        )
    except tkinter.TclError as e:
        raise SystemExit(
            f"cannot open a file dialog ({e}); use the headless options "
            "--input/--sigma/--out instead"
        ) from e

    if not file_paths:
        print("No files selected. Exiting.")
        return 0

    # Get smoothing sigma
    while True:
        try:
            sigma_smooth = float(input("Enter Gaussian smoothing sigma (positive number): "))
            if sigma_smooth <= 0:
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter a positive number.")

    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Process files
    for file_path in file_paths:
        process_image(file_path, sigma_smooth, output_dir)

    print("Processing complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
