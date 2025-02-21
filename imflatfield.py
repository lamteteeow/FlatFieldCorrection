from tkinter import filedialog
import cv2 # PIL builtin gaussian filter does not support float value
import numpy as np
import os


def estimate_flat_field(img_array, sigma=50):
    """Estimate flat field using low-frequency components"""
    # Create smoothed version for flat field estimation
    flat_estimate = cv2.GaussianBlur(img_array, (0, 0), sigmaX=sigma, sigmaY=sigma)
    print(flat_estimate)

    # Normalize to maintain average intensity
    flat_estimate = flat_estimate / np.mean(flat_estimate)
    return flat_estimate


def estimate_dark_field():
    pass


def process_image(file_path, sigma, output_dir):
    """Process image with self-contained flat field correction (16-bit grayscale implementation)"""
    try:
        # Read image with OpenCV (-1 flag for unchanged bit depth)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Could not open {os.path.basename(file_path)}")
            return

        # Check if image is 16-bit
        if img.dtype != np.uint16:
            print(f"Skipping {os.path.basename(file_path)} - not 16-bit grayscale")
            return

        # Convert to float64 preserving original values
        img_array = img.astype(np.float64)
        print(img_array)

        # Estimate flat field
        flat_field = estimate_flat_field(img_array, sigma)

        # Apply flat field correction with epsilon to prevent division by zero
        epsilon = 1e-12
        corrected = (img_array / (flat_field + epsilon)) * np.mean(flat_field)

        # Convert back to 16-bit
        # final_array = np.clip(corrected, 0, 65535).astype(np.uint16)
        final_array = corrected.astype(np.uint16)

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
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")


def main():
    # Ask user to select TIFF files
    file_paths = filedialog.askopenfilenames(
        title="Select TIFF files", filetypes=[("TIFF files", "*.tif;*.tiff")]
    )

    if not file_paths:
        print("No files selected. Exiting.")
        return

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


if __name__ == "__main__":
    main()
    print("Processing complete!")
