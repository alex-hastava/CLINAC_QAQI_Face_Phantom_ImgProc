import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_dicom(dicom_path):
    """
    Load a DICOM file and extract pixel data, slope, intercept, and windowing information.
    """
    # Load DICOM file
    dicom_data = pydicom.dcmread(dicom_path)

    # Extract necessary DICOM tags
    pixel_array = dicom_data.pixel_array  # Raw pixel data
    rescale_slope = float(dicom_data.get("RescaleSlope", 1))  # Default is 1 if not set
    rescale_intercept = float(dicom_data.get("RescaleIntercept", 0))  # Default is 0 if not set
    window_center = float(dicom_data.get("WindowCenter", np.mean(pixel_array)))  # Default: mean intensity
    window_width = float(dicom_data.get("WindowWidth", np.ptp(pixel_array)))  # Default: full range

    # Return data and the DICOM metadata
    return pixel_array, rescale_slope, rescale_intercept, window_center, window_width, dicom_data


def apply_rescale_and_windowing(pixel_array, slope, intercept, center, width):
    """
    Rescale raw pixel data using slope and intercept, then apply windowing.
    """
    # Rescale pixel values using slope and intercept
    rescaled_pixels = (pixel_array * slope) + intercept

    # Apply windowing
    window_min = center - (width / 2)
    window_max = center + (width / 2)
    windowed_pixels = np.clip(rescaled_pixels, window_min, window_max)  # Clip values outside the window
    return windowed_pixels


def plot_color_map_and_histogram_with_metadata(pixel_array, dicom_data, colormap="plasma"):
    """
    Plot the color map and histogram of pixel intensities, and display metadata.
    """
    # Normalize pixel intensities for color map visualization
    norm_pixel_array = (pixel_array - np.min(pixel_array)) / (
            np.max(pixel_array) - np.min(pixel_array))  # Normalize to 0-1

    # Extract metadata for display
    metadata = {
        "Patient ID": dicom_data.get("PatientID", "Unknown"),
        "Study Date": dicom_data.get("StudyDate", "Unknown"),
        "Modality": dicom_data.get("Modality", "Unknown"),
        "Rescale Slope": dicom_data.get("RescaleSlope", "Not Available"),
        "Rescale Intercept": dicom_data.get("RescaleIntercept", "Not Available"),
        "Window Center": dicom_data.get("WindowCenter", "Not Available"),
        "Window Width": dicom_data.get("WindowWidth", "Not Available")
    }

    # Create figure
    plt.figure(figsize=(14, 8))

    # Subplot 1: Color map
    plt.subplot(2, 2, 1)
    plt.title("Color Map (with DICOM Metadata)")
    plt.imshow(norm_pixel_array, cmap=colormap)
    plt.colorbar(label="Rescaled Intensity")
    plt.axis('off')

    # Subplot 2: Histogram
    plt.subplot(2, 2, 2)
    plt.title("Histogram of Pixel Intensities")
    plt.hist(pixel_array.flatten(), bins=256, color="blue", alpha=0.7)
    plt.xlabel("Pixel Intensity (after rescaling)")
    plt.ylabel("Frequency")

    # Subplot 3: Metadata display
    plt.subplot(2, 1, 2)
    plt.title("DICOM Metadata")
    plt.axis('off')  # Turn off axis for text box
    metadata_text = "\n".join([f"{key}: {value}" for key, value in metadata.items()])
    plt.text(0.1, 0.5, metadata_text, fontsize=12, va="center", ha="left", wrap=True)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to load, process, and visualize the DICOM file along with metadata.
    """
    dicom_path = "C:/Users/ahastava/PycharmProjects/Face_Phantom_MeV_Scan.dcm"

    # Step 1: Load DICOM file and extract parameters
    pixel_array, slope, intercept, center, width, dicom_data = load_dicom(dicom_path)

    # Step 2: Apply rescale and windowing
    processed_pixels = apply_rescale_and_windowing(pixel_array, slope, intercept, center, width)

    # Step 3: Plot color map, histogram, and metadata
    plot_color_map_and_histogram_with_metadata(processed_pixels, dicom_data)


if __name__ == "__main__":
    main()