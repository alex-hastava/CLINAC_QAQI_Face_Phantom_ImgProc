import os
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pylinac import FieldAnalysis, FieldProfileAnalysis, Protocol, Centering, Edge, Normalization, Interpolation
from pylinac.metrics.profile import (
    PenumbraLeftMetric,
    PenumbraRightMetric,
    SymmetryAreaMetric,
    FlatnessDifferenceMetric,
)
import plotly.io as pio

# Set Plotly to open in the browser
pio.renderers.default = 'browser'

# Path to DICOM image(s)
dicom_path_1 = r"C:/Users/Hasta/Downloads/Scan_1.dcm"         #Scan_1
dicom_path_2 = r"C:/Users/Hasta/Downloads/Scan_2.dcm"         #Scan_2

# Load metadata for reference
ds = pydicom.dcmread(dicom_path_1)
gantry_angle = float(ds.get('GantryAngle', 0.0))
collimator_angle = float(ds.get('BeamLimitingDeviceAngle', 0.0))
couch_angle = float(ds.get('TableAngle', 0.0))

print(f"Gantry Angle: {gantry_angle}°")
print(f"Collimator Angle: {collimator_angle}°")
print(f"Couch Angle: {couch_angle}°")

# Initialize field profile analyzer
field_analyzer = FieldProfileAnalysis(dicom_path_1)

# Run analysis with robust settings
field_analyzer.analyze(
    centering=Centering.GEOMETRIC_CENTER,
    x_width=0.02,
    y_width=0.02,
    normalization=Normalization.GEOMETRIC_CENTER,
    edge_type=Edge.INFLECTION_DERIVATIVE,
    ground=True,
    metrics=(
        PenumbraLeftMetric(),
        PenumbraRightMetric(),
        SymmetryAreaMetric(),
        FlatnessDifferenceMetric(),
    ),
)

# Print results
print("\n📊 Field Profile Analysis Results:")
print(field_analyzer.results())

# Show profile plots
field_analyzer.plot_analyzed_images(show_grid=True, mirror="beam")

# Save PDF report
try:
    field_analyzer.publish_pdf("Field_Profile_Analysis_Report.pdf")
    print("\n✅ PDF report saved as 'Field_Profile_Analysis_Report.pdf'")
except KeyError as e:
    print(f"\n⚠️ Could not generate PDF report due to missing data: {e}")

##### BEAM COMPARISON #####

ds1 = pydicom.dcmread(dicom_path_1)
ds2 = pydicom.dcmread(dicom_path_2)

# Extract pixel data
img1 = ds1.pixel_array.astype(np.float32)
img2 = ds2.pixel_array.astype(np.float32)

# Extract pixel data
img1 = ds1.pixel_array.astype(np.float32)
img2 = ds2.pixel_array.astype(np.float32)

# Normalize images for fair comparison
img1 = (img1 - img1.min()) / (img1.max() - img1.min())
img2 = (img2 - img2.min()) / (img2.max() - img2.min())

# Extract horizontal and vertical profiles through image center
center_row = img1.shape[0] // 2
center_col = img1.shape[1] // 2

profile1_x = img1[center_row, :]
profile2_x = img2[center_row, :]

profile1_y = img1[:, center_col]
profile2_y = img2[:, center_col]

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(profile1_x, label="Scan 1")
plt.plot(profile2_x, label="Scan 2", linestyle='dashed')
plt.title("Horizontal Profile (X-axis)")
plt.xlabel("Pixel Position")
plt.ylabel("Normalized Intensity")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(profile1_y, label="Scan 1")
plt.plot(profile2_y, label="Scan 2", linestyle='dashed')
plt.title("Vertical Profile (Y-axis)")
plt.xlabel("Pixel Position")
plt.ylabel("Normalized Intensity")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

my_img = FieldAnalysis(path=dicom_path_1)

my_img.analyze(
    protocol=Protocol.VARIAN,
    centering=Centering.BEAM_CENTER,
    in_field_ratio=0.8,
    is_FFF=True,
    interpolation=Interpolation.SPLINE,
    interpolation_resolution_mm=0.2,
    edge_detection_method=Edge.INFLECTION_HILL,
)

print(my_img.results())  # print results as a string
my_img.plot_analyzed_image()  # matplotlib image
my_img.publish_pdf(filename="field_analysis.pdf")  # create PDF and save to file
my_img.results_data()  # dict of results