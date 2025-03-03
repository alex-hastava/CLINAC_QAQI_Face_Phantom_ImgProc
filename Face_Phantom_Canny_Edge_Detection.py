import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def load_dicom(dicom_path):
    """ Load DICOM file, apply Gaussian smoothing, normalize, sharpen, highlight edges, and apply Canny edge detection. """
    dicom_data = pydicom.dcmread(dicom_path)
    image = dicom_data.pixel_array.astype(np.float32)

    # Step 1: Apply Gaussian Blur to reduce noise
    smoothed_image = cv2.GaussianBlur(image, (3, 3), 1)

    # Step 2: Apply the Laplacian operator for edge enhancement
    laplacian_image = cv2.Laplacian(smoothed_image, cv2.CV_32F)

    # Step 3: Convert the result to absolute values (to capture both positive and negative gradients)
    laplacian_image_abs = cv2.convertScaleAbs(laplacian_image)

    # Step 4: Normalize pixel values to 8-bit grayscale (0-255)
    normalized_image = cv2.normalize(laplacian_image_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 5: (Optional) Set pixels with value < 50 to zero
    normalized_image[normalized_image < 200] = 0

    # Step 6: Apply aggressive sharpening (stronger sharpening kernel)
    sharpen_kernel = np.array([[ -2, -3,  -2],
                               [-3,  25, -3],
                               [ -2, -3,  -2]], dtype=np.float32)
    sharpened_image = cv2.filter2D(normalized_image, -1, sharpen_kernel)

    # Step 7: Apply CLAHE (Adaptive Histogram Equalization) to enhance contrast in edge regions
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3, 3))
    enhanced_image = clahe.apply(sharpened_image)

    # Step 8: Increase the contrast further by scaling pixel values
    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=1.0, beta=0)

    # Step 10: Apply thresholding to make edges binary (only 0 or 255)
    threshold_value = 210  # Set a reasonable threshold value to detect edges
    _, thresholded_image = cv2.threshold(enhanced_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Print out pixel values of the thresholded image for debugging
    print("Pixel values of the thresholded image:")
    print(np.unique(thresholded_image))  # Should be only 0 and 255

    # Step 11: Extract contours to obtain an exact outline.
    contours, _ = cv2.findContours(thresholded_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline_image = np.zeros_like(thresholded_image)
    cv2.drawContours(outline_image, contours, -1, 255, 1)

    return image, smoothed_image, laplacian_image_abs, normalized_image, sharpened_image, enhanced_image, thresholded_image, outline_image

def main():
    # Path to DICOM file (update as needed)
    dicom_path = "C:/Users/ahastava/PycharmProjects/Face_Phantom_MeV_Scan.dcm"

    # Load DICOM and apply processing
    original_image, smoothed_image, laplacian_image, normalized_image, sharpened_image, enhanced_image, thresholded_image, outline_image = load_dicom(dicom_path)

    # Display the images step-by-step in a single figure with 8 subplots
    fig, axes = plt.subplots(1, 8, figsize=(40, 6))

    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(smoothed_image, cmap='gray')
    axes[1].set_title("Smoothed Image (Gaussian Blur)")
    axes[1].axis("off")

    axes[2].imshow(laplacian_image, cmap='gray')
    axes[2].set_title("Laplacian Image")
    axes[2].axis("off")

    axes[3].imshow(normalized_image, cmap='gray')
    axes[3].set_title("Normalized Image")
    axes[3].axis("off")

    axes[4].imshow(sharpened_image, cmap='gray')
    axes[4].set_title("Sharpened Image")
    axes[4].axis("off")

    axes[5].imshow(enhanced_image, cmap='gray')
    axes[5].set_title("Enhanced Image (CLAHE)")
    axes[5].axis("off")

    axes[6].imshow(thresholded_image, cmap='gray')
    axes[6].set_title("Thresholded Edge Detection")
    axes[6].axis("off")

    axes[7].imshow(outline_image, cmap='gray')
    axes[7].set_title("Exact Outline")
    axes[7].axis("off")

    plt.tight_layout()
    plt.show()

    # (Additional code for histograms and saving the image can follow here)

if __name__ == "__main__":
    main()
