import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib

# Ensure compatibility with PyCharm
matplotlib.use("TkAgg")


def load_dicom(dicom_path):
    """ Load DICOM file and extract metadata. """
    dicom_data = pydicom.dcmread(dicom_path)
    image = dicom_data.pixel_array.astype(np.float32)  # Ensure float for processing

    # Normalize pixel values to 8-bit grayscale (0-255)
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Extract key DICOM metadata
    metadata = {
        "Modality": dicom_data.get("Modality", "Unknown"),
        "Intercept": dicom_data.get("RescaleIntercept", 0),
        "Slope": dicom_data.get("RescaleSlope", 1),
        "Window": f"{dicom_data.get('WindowCenter', 'N/A')} / {dicom_data.get('WindowWidth', 'N/A')}",
        "Intensity Relation": dicom_data.get((0x0028, 0x1040), "Unknown"),
        "Sign": dicom_data.get((0x0028, 0x1041), 1),  # Extract pixel intensity relationship sign (numerical value)
    }

    return normalized_image, metadata


def update_plot(val):
    """ Update histogram and row indicator in the DICOM image """
    row_index = int(val)  # Selected row
    row_values = dicom_image[row_index, :]

    # Update histogram
    ax_hist.clear()
    ax_hist.plot(row_values, color='blue', linewidth=1)
    ax_hist.set_title(f"Pixel Intensity (Row {row_index})")
    ax_hist.set_xlabel("Pixel Position")
    ax_hist.set_ylabel("Intensity (0-255)")
    ax_hist.grid(True)

    # Update row indicator in the DICOM image
    ax_img.clear()
    ax_img.imshow(dicom_image, cmap="gray", aspect="auto")
    ax_img.set_title("DICOM Image")
    ax_img.axhline(row_index, color='red', linestyle='--', linewidth=1)  # Red line indicator
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # Interpret pixel intensity relationship
    sign_desc = "Higher = More radiation" if metadata["Sign"] == 1 else "Higher = Less radiation"

    # Update metadata display
    ax_meta.clear()
    ax_meta.axis("off")
    ax_meta.set_title("Metadata", fontsize=12)

    # Updated multiline text with pixel relationship
    text = "\n".join([
        f"Modality: {metadata['Modality']}",
        f"Intercept / Slope: {metadata['Intercept']} / {metadata['Slope']}",
        f"Window: {metadata['Window']}",
        #f"Intensity Relation: ",
        #f"{metadata['Intensity Relation']}",
        f"{metadata['Sign']}",
        f"{sign_desc}",
    ])

    # Display text on the metadata panel
    ax_meta.text(0, 0.9, text, fontsize=10, verticalalignment="top", horizontalalignment="left",
                 transform=ax_meta.transAxes)

    fig.canvas.draw_idle()


def main():
    global dicom_image, metadata, fig, ax_img, ax_hist, ax_meta, slider

    # Path to DICOM file
    dicom_path = "C:/Users/ahastava/PycharmProjects/Face_Phantom_MeV_Scan.dcm"

    # Load DICOM image and metadata
    dicom_image, metadata = load_dicom(dicom_path)

    # Create figure with three sections (DICOM image, histogram, metadata)
    fig, (ax_img, ax_hist, ax_meta) = plt.subplots(1, 3, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 3, 1]})
    # Here, the first two panels are wider (3 parts each), and the last one (metadata) is smaller (1 part).


    plt.subplots_adjust(bottom=0.25)  # Space for slider

    # Add slider for row selection
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Row', 0, dicom_image.shape[0] - 1, valinit=dicom_image.shape[0] // 2, valstep=1)

    # Connect slider to update function
    slider.on_changed(update_plot)

    # Initialize plot
    update_plot(slider.val)

    plt.show(block=True)


if __name__ == "__main__":
    main()
