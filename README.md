# Flat field correction

This script automatically estimates **16-bit grayscale** TIFF format images and corrects them to remove some of vignette effect using **opencv** implementation of **Gaussian filter**

Unfortunately the Gaussian image filter implementation in Pillow does not support type float till this day, I am using OpenCV instead and subjected to their License term.

**HINT:** region of interest **sigma** should be around 10-30

**TODO:**

- [ ] Implement dark field references as well
- [ ] Reimplement Gaussian blur using Pillow because of OpenCV license

**DISCLAIMER:** I am not sure about this Apache 2.0 license and therefore please use at your own risk.
