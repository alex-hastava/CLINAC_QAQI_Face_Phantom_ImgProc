import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_dicom(dicom_path, save_path="contour_output.png"):
    """Load DICOM, enhance contrast, extract cutoff contours."""
    # Load DICOM image
    dicom_data = pydicom.dcmread(dicom_path)
    image = dicom_data.pixel_array.astype(np.float32)

    # Step 1: Normalize image (0-255)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 2: Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(50, 50))
    enhanced_image = clahe.apply(image)

    # Step 3: Apply Gaussian Blur (preserve structures while reducing noise)
    blurred_image = cv2.GaussianBlur(enhanced_image, (7, 7), 0)

    # Step 4: Edge Detection using Adaptive Canny
    median_val = np.median(blurred_image)
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(blurred_image, lower, upper)

    # Step 5: Adaptive Thresholding (Highlight collimator/light field)
    adaptive_thresh = cv2.adaptiveThreshold(
        enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 7, 7)

    # Step 6: Combine Canny edges and thresholded image
    combined_edges = cv2.bitwise_or(adaptive_thresh, edges)

    # Step 7: Morphological Closing (Fill small gaps)
    kernel = np.ones((2, 2), np.uint8)
    morphed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)

    # Step 8: Masking the light field (removing noise inside the light field)
    _, light_field_mask = cv2.threshold(enhanced_image, 35, 255, cv2.THRESH_BINARY)

    # Remove light field area from the processed edges
    morphed[light_field_mask == 255] = 0  # Set the light field region to black (no edges)

    # Step 9: Set all non-zero values in morphed to 255
    morphed[morphed > 0] = 255  # Set any value > 0 to 255
    morphed[morphed == 0] = 0  # Keep 0s as 0

    # Debug Step: Ensure contours exist and show the combined edges plot
    plt.figure(figsize=(6, 6))
    plt.imshow(morphed, cmap='gray')
    plt.title("Processed Edges (No Light Field Noise)")
    plt.axis("off")
    plt.show()

    # Return the processed image before inversion
    return image, enhanced_image, combined_edges, morphed


def convert_to_color(morphed_image):
    """Convert the morphed grayscale image back to 32-bit color format."""
    # Convert morphed (grayscale) to 3 channels (RGB) so color can be visualized
    color_image = cv2.cvtColor(morphed_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Convert to 32-bit (float32) format to see the colors clearly
    color_image = color_image.astype(np.float32) / 255.0  # Normalize to range [0, 1]

    return color_image


def detect_circles_and_rectangle(morphed_image, color_morphed_image, dicom_data):
    """Detect circles using Hough Transform and draw a rectangle around the light field."""

    # Extract PixelSpacing from the DICOM metadata
    # Extract PixelSpacing from the DICOM metadata using the correct tag (3002,0011)

    pixel_spacing = dicom_data.get((0x3002, 0x0011))  # Tag (3002, 0011)
    # Assuming the spacing is the same for both row and column
    pixel_spacing = float(pixel_spacing[0])  # Use the first value for both row and column spacing

    # Now pixel_spacing will be a list of [row_spacing, col_spacing] in mm/pixel
    print(f"Pixel Spacing (Rows/Cols): {pixel_spacing:.3f} mm")

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        morphed_image,
        cv2.HOUGH_GRADIENT,
        dp=1.4,  # Inverse ratio of accumulator resolution to image resolution
        minDist=20,  # Minimum distance between circle centers
        param1=130,  # Higher threshold for edge detection
        param2=30,  # Accumulator threshold for circle detection
        minRadius=20,  # Minimum circle radius
        maxRadius=60  # Maximum circle radius
    )

    image_size = color_morphed_image.shape[0]  # Number of pixels in one dimension

    # Initialize an empty list to store the circle data (offset_x, offset_y, radius_mm)
    circle_data = []

    # Draw the detected circles in color (on the color version of the image)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Compute the scaled image center in mm
        image_center_x_mm = (2 / 3) * ((image_size / 2) * pixel_spacing)
        image_center_y_mm = (2 / 3) * ((image_size / 2) * pixel_spacing)

        for (x, y, r) in circles:
            # Convert pixel coordinates to mm and scale to 2/3
            x_mm = (2 / 3) * (x * pixel_spacing)
            y_mm = (2 / 3) * (y * pixel_spacing)

            # Convert radius from pixels to mm and scale to 2/3 SAD
            radius_mm = (2 / 3) * (r * pixel_spacing)
            diameter_mm = 2 * radius_mm  # Directly use scaled radius for diameter

            # Print the converted coordinates and diameter
            print(
                f"Circle at ({x_mm:.2f} mm, {y_mm:.2f} mm) - Radius: {radius_mm:.2f} mm, Diameter: {diameter_mm:.2f} mm")

            # Draw circle in red
            cv2.circle(color_morphed_image, (x, y), r, (255, 0, 0), 4)

            # Calculate the offsets from the scaled beam center (in mm)
            offset_x = x_mm - image_center_x_mm
            offset_y = y_mm - image_center_y_mm

            # Print the offsets (in mm)
            print(f"Offset from beam center: X = {offset_x:.2f} mm, Y = {offset_y:.2f} mm")

            # Display the diameter in mm at the center of the circle
            distance_text = f"{diameter_mm:.2f}"  # Show diameter with 2 decimal places
            cv2.putText(color_morphed_image, distance_text, (x - 20, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                        2)

            # Save the circle's data (offset_x, offset_y, radius_mm)
            circle_data.append([offset_x, offset_y, radius_mm])

    # Save the circle data (offsets and radius) to a .npy file
    np.save("my_special_phantom_bbs.npy", circle_data)  # Save offsets and radius data

    distance_unit = f"Units: mm (diameter)"
    cv2.putText(color_morphed_image, distance_unit, (450, 1100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # Find the contours and draw a rectangle around the light field (based on intensity mask)
    contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(color_morphed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle in green

    # Debug Step: Show the final output with circles and rectangle
    plt.figure(figsize=(6, 6))
    plt.imshow(color_morphed_image)
    plt.title("Circles Detected and Rectangle around Light Field")
    plt.axis("off")
    plt.show()


def main():
    # Rad_Onc_PC
    dicom_path = "C:/Users/ahastava/PycharmProjects/Face_Phantom_MeV_Scan.dcm"
    save_path = "C:/Users/ahastava/PycharmProjects/contour_output.png"

    # Process DICOM image
    dicom_data = pydicom.dcmread(dicom_path)  # Load the DICOM data
    original_image, enhanced_image, combined_edges, morphed_image = load_dicom(dicom_path, save_path)

    # Convert the processed grayscale morphed image to color format for visualization
    color_morphed_image = convert_to_color(morphed_image)

    # Detect circles and rectangles on the morphed image (in color)
    detect_circles_and_rectangle(morphed_image, color_morphed_image, dicom_data)

    # Save the final 8-bit grayscale output (morphed image without color)
    cv2.imwrite(save_path, morphed_image)  # Save the 8-bit grayscale (processed) image
    print(f"Final output saved to {save_path}")

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    titles = ["Original", "Enhanced", "Processed Edges with Circles and Rectangle"]
    images = [original_image, enhanced_image, color_morphed_image]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
