"""View-only 8-bit previews of 16-bit frames, with ImageJ's display windows.

The frame data is 16-bit but only occupies a small part of the 0-65535 range,
so viewers that map the full range render it near-black.  This renders an
8-bit PNG for *looking at* the data -- it is never an analysis artifact; the
16-bit TIFFs stay the data of record.

Three windowing modes, the first two ported from ImageJ's own source:

``auto``    the Brightness/Contrast "Auto" button, i.e.
            ``ij.plugin.frame.ContrastAdjuster.autoAdjust()``:
            256 bins spanning the image's own min..max (ShortStatistics uses
            ``binSize = (dataMax-dataMin+1)/256``), any bin holding more than
            10% of the pixels ignored (it is background, not signal), and the
            first bin from each end holding more than ``pixelCount/5000``
            pixels sets the range.  Pressing Auto again halves the static
            ``autoThreshold`` field (5000 -> 2500 -> ...), which doubles that
            per-bin threshold, so the window NARROWS and more pixels saturate --
            as documented for repeated use of Auto.

``enhance`` ``Process > Enhance Contrast`` with the default ``saturated=0.35``,
            i.e. ``ij.plugin.ContrastEnhancer.getMinAndMax()`` on the exact
            16-bit histogram: walk in from both ends until the *cumulative*
            count exceeds ``pixelCount*saturated/200`` (0.175% of the pixels at
            each end), so a hot pixel can never set the range.

``fixed``   an explicit ``--window-min``/``--window-max`` (comparable between
            frames, e.g. raw vs corrected).

``batch``   one window for the whole input, from the pooled histogram -- the
            same walk as ``enhance`` over all frames' counts, i.e. ImageJ's
            "Enhance Contrast" with "Use stack histogram" on a 16-bit stack.
            Use this to flip through a sequence without the brightness
            changing per frame.

Note on ImageJ versions (why a port can differ by a fraction of a bin): the
grid the 'auto' mode walks comes from ``ShortStatistics``' ``binSize``, which is
``(histMax-histMin)/nBins`` up to ImageJ 1.54j and
``(histMax-histMin+1)/nBins`` from 1.54r/master on.  ``ContrastAdjuster``'s own
code is identical across those versions.  This module uses the current formula.

Usage:
    uv run frame_preview.py --input FRAMES_DIR --out OUT_DIR --mode auto
    uv run frame_preview.py --input FRAMES_DIR --out OUT_DIR --mode batch
    uv run frame_preview.py --input frame.tif --mode auto --print-window
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

TIFF_SUFFIXES = (".tif", ".tiff")
UINT16_VALUES = 65536
AUTO_THRESHOLD = 5000  # ContrastAdjuster.AUTO_THRESHOLD
ENHANCE_SATURATED = 0.35  # ContrastEnhancer.defaultSaturated


def _per_value_histogram(image: np.ndarray) -> np.ndarray:
    return np.bincount(image.ravel(), minlength=UINT16_VALUES)


def imagej_auto_window(
    image: np.ndarray, auto_threshold: int = AUTO_THRESHOLD, bin_size_plus_one: bool = True
):
    """Port of ImageJ's Brightness/Contrast "Auto" (ContrastAdjuster.autoAdjust).

    ``auto_threshold`` is ``ContrastAdjuster.autoThreshold``: 5000 on the first
    click, halved on every further click on the same image.  ``bin_size_plus_one``
    selects ``ShortStatistics``' bin formula: True for ImageJ 1.54r and later
    (``(histMax-histMin+1)/nBins``), False for 1.54j and earlier
    (``(histMax-histMin)/nBins``) -- the two differ by well under one bin.
    """
    flat = image.ravel()
    pixel_count = flat.size
    # ShortStatistics.getRawMinAndMax(): first/last non-empty value -> the
    # min/max that span the statistic's histogram grid.
    nonempty = np.flatnonzero(_per_value_histogram(image))
    data_min, data_max = int(nonempty[0]), int(nonempty[-1])
    # ShortStatistics.getStatistics(): nBins=256 bins over that span.
    n_bins = 256
    span = data_max - data_min + (1 if bin_size_plus_one else 0)
    bin_size = span / n_bins
    index = np.minimum(((flat - data_min) / bin_size).astype(np.int64), n_bins - 1)
    histogram = np.bincount(index, minlength=n_bins)

    limit = pixel_count // 10  # a bin this full is background, not signal
    threshold = pixel_count // auto_threshold
    qualifying = np.where(histogram > limit, 0, histogram) > threshold
    hits = np.flatnonzero(qualifying)
    h_min = int(hits[0]) if hits.size else n_bins - 1
    h_max = int(hits[-1]) if hits.size else 0

    low = data_min + h_min * bin_size
    high = data_min + h_max * bin_size
    if low == high:  # ImageJ falls back to the data range
        low, high = float(data_min), float(data_max)
    return float(low), float(high)


def _enhance_walk(histogram: np.ndarray, saturated: float):
    """ContrastEnhancer.getMinAndMax() walk over a (possibly pooled) histogram."""
    threshold = int(histogram.sum() * saturated / 200.0)
    from_low = np.flatnonzero(np.cumsum(histogram) > threshold)
    from_high = np.flatnonzero(np.cumsum(histogram[::-1]) > threshold)
    low = int(from_low[0]) if from_low.size else UINT16_VALUES - 1
    high = UINT16_VALUES - 1 - int(from_high[0]) if from_high.size else 0
    return float(low), float(high)


def imagej_enhance_window(image: np.ndarray, saturated: float = ENHANCE_SATURATED):
    """Port of ImageJ's Process > Enhance Contrast (ContrastEnhancer.getMinAndMax)."""
    return _enhance_walk(_per_value_histogram(image), saturated)


def imagej_batch_window(histogram: np.ndarray, saturated: float = ENHANCE_SATURATED):
    """One window for a whole sequence, from the pooled histogram.

    This is ImageJ's ``Enhance Contrast`` with "Use stack histogram" on a 16-bit
    stack: ``StackStatistics`` sums the 65536-bin histograms of all slices and
    the same walk runs over the total, so every preview gets an identical,
    flicker-free window and no frame can drag the brightness of the others.
    """
    return _enhance_walk(histogram, saturated)


def resolve_window(
    image: np.ndarray,
    mode: str,
    window_min: float,
    window_max: float,
    auto_threshold: int,
    bin_size_plus_one: bool,
):
    if mode == "auto":
        return imagej_auto_window(image, auto_threshold, bin_size_plus_one)
    if mode == "enhance":
        return imagej_enhance_window(image)
    return window_min, window_max


def read_frame(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2 or image.dtype != np.uint16:
        raise SystemExit(f"expected a 2D 16-bit frame: {path}")
    return image


def pooled_histogram(frames: list[Path]) -> np.ndarray:
    histogram = np.zeros(UINT16_VALUES, dtype=np.int64)
    for path in frames:
        histogram += _per_value_histogram(read_frame(path))
    return histogram


def to_uint8(image: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = (image.astype(np.float32) - low) / max(high - low, 1e-9) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def collect_frames(inputs: list[Path]) -> list[Path]:
    frames: list[Path] = []
    for item in inputs:
        path = item.expanduser()
        if path.is_dir():
            found = sorted(
                child
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in TIFF_SUFFIXES
            )
            if not found:
                raise SystemExit(f"no TIFF files in directory: {path}")
            frames.extend(found)
        elif path.is_file():
            frames.append(path)
        else:
            raise SystemExit(f"no such file or directory: {path}")
    return frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="TIFF files and/or directories")
    parser.add_argument("--out", type=Path, help="output directory for the PNG previews")
    parser.add_argument("--mode", choices=("auto", "enhance", "batch", "fixed"), default="auto")
    parser.add_argument("--window-min", type=float, default=0.0, help="fixed mode: lower bound")
    parser.add_argument("--window-max", type=float, default=65535.0, help="fixed mode: upper bound")
    parser.add_argument("--auto-threshold", type=int, default=AUTO_THRESHOLD, help="auto mode: ContrastAdjuster.autoThreshold (5000, halve per extra Auto click)")
    parser.add_argument(
        "--bin-formula",
        choices=("plus-one", "no-plus-one"),
        default="plus-one",
        help="auto mode: ShortStatistics binSize formula. plus-one = ImageJ 1.54r+ "
        "(current, default); no-plus-one = ImageJ <=1.54j. They differ by <1 bin.",
    )
    parser.add_argument("--print-window", action="store_true", help="print the display window per frame and write no PNGs")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.print_window and args.out is None:
        raise SystemExit("--out is required unless --print-window is given")
    frames = collect_frames(args.input)
    if args.out is not None:
        args.out.expanduser().mkdir(parents=True, exist_ok=True)

    # "batch" needs every frame before it can window any of them: pool the
    # histograms (bounded memory), then write with that one window.
    batch_window = None
    if args.mode == "batch":
        batch_window = imagej_batch_window(pooled_histogram(frames))
        if args.print_window:
            print(f"mode=batch frames={len(frames)} window={batch_window[0]:.3f}..{batch_window[1]:.3f}")
            return 0
        print(f"batch window from {len(frames)} frames: {batch_window[0]:.1f}..{batch_window[1]:.1f}")

    for path in frames:
        image = read_frame(path)
        low, high = (
            batch_window
            if batch_window is not None
            else resolve_window(
                image,
                args.mode,
                args.window_min,
                args.window_max,
                args.auto_threshold,
                args.bin_formula == "plus-one",
            )
        )
        if args.print_window:
            print(f"{path.name} mode={args.mode} window={low:.3f}..{high:.3f}")
            continue
        target = args.out.expanduser() / f"{path.stem}_prev.png"
        if not cv2.imwrite(str(target), to_uint8(image, low, high)):
            raise SystemExit(f"could not write {target}")
        print(f"{target.name}  window={low:.1f}..{high:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
