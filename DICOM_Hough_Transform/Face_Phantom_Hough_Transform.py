import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Load the DICOM file
def load_dicom(filepath):
    dicom_file = pydicom.dcmread(filepath)
    image_array = dicom_file.pixel_array
    return image_array


# Preprocess the image
def preprocess_image(image):
    # Normalize image intensity to the range 0-255 for OpenCV
    image = image.astype(np.float32)
    image -= np.min(image)
    image /= np.max(image)
    image = (image * 255).astype(np.uint8)
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred


# Hough Circle Transform
def detect_circles(image):
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # Inverse ratio of accumulator resolution to image resolution
        minDist=20,  # Minimum distance between circle centers
        param1=20,  # Higher threshold for edge detection
        param2=40,  # Accumulator threshold for circle detection
        minRadius=1,  # Minimum circle radius
        maxRadius=50  # Maximum circle radius
    )
    return circles


# Display results
def display_results(original_image, processed_image, circles):
    # Convert the original DICOM image to an 8-bit grayscale format for display
    original_image = original_image.astype(np.float32)
    original_image -= np.min(original_image)
    original_image /= np.max(original_image)
    original_image = (original_image * 255).astype(np.uint8)

    # Display images
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    ax[0].imshow(original_image, cmap="gray")
    ax[0].set_title("Original DICOM Image", y=0.2)
    ax[0].axis("off")

    # Circles overlayed on the original image
    for circle in circles[0, :]:
        center = (int(circle[0]), int(circle[1]))  # Circle center
        radius = int(circle[2])  # Circle radius
        cv2.circle(original_image, center, radius, (255, 0, 0), 2)
        cv2.circle(original_image, center, 2, (0, 255, 0), 3)  # Small dot at circle center

    ax[1].imshow(original_image, cmap="gray")
    ax[1].set_title("Detected Circles", y=0.2)
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


# Main Function
def main():
    #dicom_path = "C:/Users/Hasta/OneDrive/Documents/!!SBU BME (B.E.-M.S.)/Zhang Lab (Rad Onc)/Face_Phantom_6MeV_Scan.dcm"  # Personal Computer DICOM file path.
    dicom_path = "C:/Users/ahastava/PycharmProjects/Face_Phantom_Hough_Circle_Transform/Face_Phantom_6MeV_Scan.dcm"  # SBUH Medical Physics Computer DICOM file path.
    image = load_dicom(dicom_path)
    processed_image = preprocess_image(image)
    circles = detect_circles(processed_image)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        display_results(image, processed_image, circles)
    else:
        print("No circles detected!")


if __name__ == "__main__":
    main()