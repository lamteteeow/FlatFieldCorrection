# Flat field correction

This script automatically estimates **16-bit grayscale** TIFF format images and corrects them to remove some of vignette effect using **opencv** implementation of **Gaussian filter**

Unfortunately the Gaussian image filter implementation in Pillow does not support type float till this day, I am using OpenCV instead and subjected to their License term.

## Usage

Please use [uv](https://docs.astral.sh/uv/) to manage python packages/dependencies/versions/... It's 2025 guys.

1. Making sure dependencies are installed:

    ```bash
        uv sync
    ```

2. or for under-the-rock people:

    ```bash
        pip install -r requirements
    ```

3. Run:

    ```bash
        uv run imflatfield.py
    ```

    The script then opens a file dialog and asks for **sigma**.

4. Or **without the file dialog** (batch/headless, e.g. over a whole dataset):

    ```bash
        uv run imflatfield.py \
            --input /work/cf_lam/pipeline/datasets/HV0371 \
            --sigma 20 \
            --out /work/cf_lam/imflatfield/HV0371
    ```

    `--input` accepts TIFF files and/or directories (scanned non-recursively for
    `*.tif`/`*.tiff`), `--out` receives the corrected frames under the same file
    names.  Existing outputs are skipped unless `--overwrite` is passed, frames are
    processed by `--workers` parallel processes, and the command exits non-zero if
    any frame fails.  Outputs are uncompressed 16-bit grayscale TIFF.  If corrected
    values exceed 65535 (possible in near-black regions, where the estimated field
    is small) the cast wraps them around; `--clip` saturates instead and the command
    warns about it either way.

**HINT:** region of interest **sigma** should be around 10-30

5. To *look* at 16-bit frames (including the corrected ones) outside ImageJ/Fiji:

    ```bash
        uv run frame_preview.py \
            --input /work/cf_lam/imflatfield/HV0371 \
            --out   /work/cf_lam/imflatfield/_preview \
            --mode  auto
    ```

    `frame_preview.py` writes **view-only** 8-bit PNGs (`<frame>_prev.png`); the 16-bit
    TIFFs remain the data of record.  The frame values occupy only a small part
    of 0-65535, so viewers that map the full range (Windows Photos, Explorer
    thumbnails, `cv2.imread` without `IMREAD_UNCHANGED`) show them near-black.

    `--mode` selects the display window and the first two are ported from
    ImageJ's own source, so they match what Fiji shows:

    - `auto`    (default) the Brightness/Contrast **Auto** button —
      `ij.plugin.frame.ContrastAdjuster.autoAdjust()`: 256 bins spanning the
      image's own min..max (`binSize=(dataMax-dataMin+1)/256`), bins holding
      more than 10% of the pixels ignored as background, and the first bin from
      each end holding more than `pixelCount/5000` pixels setting the range.
      `--auto-threshold N` mirrors ImageJ's `autoThreshold` field: 5000 on the
      first click and halved per further click, which narrows the window.
    - `enhance` `Process > Enhance Contrast` with ImageJ's default
      `saturated=0.35` — walks the exact 16-bit histogram in from both ends
      until 0.175% of the pixels are cut at each side.  Immune to isolated
      bright pixels (its grid is per-value, not data-range dependent).
    - `fixed`   an explicit `--window-min`/`--window-max`, e.g. for raw-vs-
      corrected comparisons in one common window.
    - `batch`   one window for the whole input, from the pooled histogram of all
      frames (ImageJ's "Enhance Contrast" with "Use stack histogram" on a 16-bit
      stack).  Per-frame windows drift a few percent with the acquisition, so use
      this to flip through a sequence without the brightness changing per frame.

    Add `--print-window` to print the window per frame without writing PNGs.
    ImageJ versions differ by one character in the grid `auto` walks: `--bin-formula
    no-plus-one` matches ImageJ ≤ 1.54j, the default `plus-one` matches 1.54r and
    later; the two are under one bin apart.

**TODO:**

- [ ] Implement dark field references as well
- [ ] Reimplement Gaussian blur using Pillow because of OpenCV license

**DISCLAIMER:** I am not sure about this Apache 2.0 license and therefore please use at your own risk.
