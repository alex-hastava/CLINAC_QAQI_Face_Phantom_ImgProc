import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_dicom(dicom_path):
    """Load DICOM file and normalize pixel data."""
    dicom_data = pydicom.dcmread(dicom_path)
    image = dicom_data.pixel_array  # Extract pixel data as numpy array
    # Normalize to 8-bit grayscale (0 to 255)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def detect_boundaries(image):
    """
    Detect boundaries of radiation fields and phantoms using edge detection
    and contour finding.
    """
    # 1. Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), sigmaX=1.5)

    # 2. Use Canny Edge Detection
    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

    # 3. Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Filter contours by size or shape if needed (e.g., remove small or irrelevant artifacts)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter out very small contours with area less than 500 pixels
            filtered_contours.append(contour)

    # 5. Create a blank mask to draw the contours
    mask = np.zeros_like(image)
    cv2.drawContours(mask, filtered_contours, -1, (255, 255, 255), thickness=2)  # Draw contours on the mask

    return mask, filtered_contours  # Mask with boundaries and the extracted contours


def main():
    # Path to the DICOM file
    dicom_path = "C:/Users/ahastava/PycharmProjects/Face_Phantom_MeV_Scan.dcm"

    # Step 1: Load DICOM file
    dicom_image = load_dicom(dicom_path)

    # Step 2: Detect boundaries of radiation fields and phantoms
    boundaries, contours = detect_boundaries(dicom_image)

    # Step 3: Display the original image and detected boundaries side by side
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original DICOM Image")
    plt.imshow(dicom_image, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Detected Boundaries")
    plt.imshow(boundaries, cmap="gray")

    plt.show()


if __name__ == "__main__":
    main()