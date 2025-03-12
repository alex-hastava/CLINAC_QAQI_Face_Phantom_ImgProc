import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylinac import Protocol, Centering, Edge, Normalization, Interpolation
from pylinac import FieldProfileAnalysis, FieldAnalysis, Device, Centering, Normalization, Edge
from pylinac.metrics.profile import (
    PenumbraLeftMetric,
    PenumbraRightMetric,
    SymmetryAreaMetric,
    FlatnessDifferenceMetric,
)
from pylinac import WinstonLutzMultiTargetMultiField
from collections import namedtuple

# Rad_Onc_PC
dicom_path = r"C:/Users/ahastava/PycharmProjects/Face_Phantom_MeV_Scan.dcm"
wl_path = r"C:/Users/ahastava/PycharmProjects/CLINAC_QAQI_Face_Phantom_ImgProc/wl_images"

#########################################################################################
# Perform field profile analysis

field_analyzer = FieldProfileAnalysis(dicom_path)
field_analyzer.analyze(
    centering=Centering.GEOMETRIC_CENTER,
    x_width=0.02,
    y_width=0.02,
    normalization=Normalization.MAX,
    edge_type=Edge.INFLECTION_HILL,
    hill_window_ratio=0.05,
    ground=True,
    metrics=(
        PenumbraLeftMetric(),
        PenumbraRightMetric(),
        SymmetryAreaMetric(),
        FlatnessDifferenceMetric(),
    ),
)

# Plot analyzed field images
field_analyzer.plot_analyzed_images(show_grid=True, mirror="beam")

# Get results from field analyzer
results = field_analyzer.results_data()
print("Field Analysis Results:", results)

# Load detected BBs from Face Phantom Canny Edge Detection
circles = np.load("bb_pixel_coords.npy")  # Ensure this file is generated beforehand
bb_offsets = [(int(x), int(y), 2 * r) for x, y, r in circles]
print("Loaded BB Data:", bb_offsets)

my_img = FieldAnalysis(path=dicom_path)
my_img.analyze(protocol=Protocol.VARIAN, centering=Centering.BEAM_CENTER, in_field_ratio=0.8,
               is_FFF=True, interpolation=Interpolation.SPLINE, interpolation_resolution_mm=0.2,
               edge_detection_method=Edge.INFLECTION_HILL)

print(my_img.results())  # print results as a string
my_img.plot_analyzed_image()  # matplotlib image
my_img.publish_pdf(filename="flatsym1.pdf")  # create PDF and save to file
my_img.results_data()  # dict of results

#########################################################################################
# Perform Winston-Lutz analysis

# Load BB configuration data from .npy file
file_path = "my_special_phantom_bbs.npy"
loaded_bbs = np.load(file_path, allow_pickle=True)

print(loaded_bbs)

# Define BB structure to include the offsets and radius size
BB = namedtuple("BB", ["offset_left_mm", "offset_up_mm", "offset_in_mm", "rad_size_mm"])

# Organize BBs into the arrangement based on their relationships
arrange = [
    BB(offset_left_mm=bb[0],    # Assuming values are already in mm
       offset_up_mm=bb[1],      # Assuming values are already in mm
       offset_in_mm=0.0,        # Assuming no in/out offsets are provided, set to 0
       rad_size_mm=bb[2])       # Assuming bb[2] is already in mm
    for i, bb in enumerate(loaded_bbs, start=1)
]

print(arrange)

# Initialize a dictionary to store the results for each edge
results = {}

# Define radiation field positions (CAX to edge distances)
radiation_field_positions = {
    "bottom": 73.9,
    "right": 74.5,
    "left": 75.4,
    "top": 74.7
}

# Define penumbra gradients (percentage per mm)
penumbra_gradients = {
    "bottom": 24.36,
    "right": 26.34,
    "left": 25.87,
    "top": 24.22
}

# Define penumbra widths (mm)
penumbra_widths = {
    "bottom": 3.0,
    "right": 2.7,
    "left": 2.7,
    "top": 2.9
}

# Determine the edge with the highest penumbra gradient
highest_gradient_edge = max(penumbra_gradients, key=penumbra_gradients.get)

# Use the maximum gradient edge's position as the radiation field reference
radiation_field_position = radiation_field_positions[highest_gradient_edge]

# Adjust radiation field position based on penumbra width
if highest_gradient_edge in ["left", "top"]:
    radiation_field_position_with_penumbra = radiation_field_position - penumbra_widths[highest_gradient_edge]
else:
    radiation_field_position_with_penumbra = radiation_field_position + penumbra_widths[highest_gradient_edge]

# Calculate Light Field Positions (based on circle locations)
# Bottom field (BB#1, BB#2) -> offset_up_mm + 15mm
light_field_bottom = ((arrange[0].offset_up_mm + arrange[1].offset_up_mm) / 2) + 15

# Right field (BB#3, BB#4) -> offset_left_mm + 15mm
light_field_right = ((arrange[2].offset_left_mm + arrange[3].offset_left_mm) / 2) + 15

# Left field (BB#5, BB#7) -> offset_left_mm - 15mm
light_field_left = ((arrange[4].offset_left_mm + arrange[6].offset_left_mm) / 2) - 15

# Top field (BB#6, BB#8) -> offset_up_mm - 15mm
light_field_top = ((arrange[5].offset_up_mm + arrange[7].offset_up_mm) / 2) - 15

# Store results for each edge
results = {
    "bottom": {
        "light_field_position":abs(light_field_bottom),
        "radiation_field_position": radiation_field_position_with_penumbra,
        "distance_between_fields": abs(light_field_bottom - radiation_field_position_with_penumbra)
    },
    "right": {
        "light_field_position": abs(light_field_right),
        "radiation_field_position": radiation_field_position_with_penumbra,
        "distance_between_fields": abs(light_field_right - radiation_field_position_with_penumbra)
    },
    "left": {
        "light_field_position": abs(light_field_left),
        "radiation_field_position": radiation_field_position_with_penumbra,
        "distance_between_fields": abs(light_field_left + radiation_field_position_with_penumbra)
    },
    "top": {
        "light_field_position": abs(light_field_top),
        "radiation_field_position": radiation_field_position_with_penumbra,
        "distance_between_fields": abs(light_field_top + radiation_field_position_with_penumbra)
    }
}

# Print results for all four edges
for edge, result in results.items():
    print(f"Results for {edge.capitalize()} edge:")
    print(f"  Light Field Position: {result['light_field_position']:.2f} mm")
    print(f"  Radiation Field Position (CAX distance + Penumbra): {result['radiation_field_position']:.2f} mm")
    print(f"  Distance between Light and Radiation Fields: {result['distance_between_fields']:.2f} mm\n")


# Define radiation field center distances (in mm) from the previous task
CAX_to_Left_edge = 75.4
CAX_to_Right_edge = 74.5
CAX_to_Top_edge = 74.7
CAX_to_Bottom_edge = 73.9

# Calculate the radiation field center (midpoint of the four edges)
radiation_field_center_x = (CAX_to_Left_edge + CAX_to_Right_edge) / 2
radiation_field_center_y = (CAX_to_Top_edge + CAX_to_Bottom_edge) / 2

# Calculate the crosshair center (average of light field positions)
crosshair_center_x = (results['left']['light_field_position'] + results['right']['light_field_position']) / 2
crosshair_center_y = (results['top']['light_field_position'] + results['bottom']['light_field_position']) / 2

# Print Crosshair Center position
print(f"Crosshair Center Position:")
print(f"  X: {crosshair_center_x:.2f} mm")
print(f"  Y: {crosshair_center_y:.2f} mm")

# Calculate and print the distance between the crosshair center and the radiation field center
distance_to_radiation_field_center = np.sqrt(
    (crosshair_center_x - radiation_field_center_x)**2 +
    (crosshair_center_y - radiation_field_center_y)**2
)

print(f"Distance between Crosshair Center and Radiation Field Center: {distance_to_radiation_field_center:.2f} mm")
print("\n")

# Define the acceptable QA tolerance (in mm)
qa_tolerance = 2.0  # 1-2 mm as specified

# Check if the distance is within the tolerance
if distance_to_radiation_field_center <= qa_tolerance:
    print(f"QA Check: The crosshair center is within the acceptable tolerance of {qa_tolerance} mm.")
else:
    print(f"QA Check: The crosshair center is OUTSIDE the acceptable tolerance of {qa_tolerance} mm.")


dicom_data = pydicom.dcmread(dicom_path)

# Get pixel array from DICOM file
img_array = dicom_data.pixel_array

# Extract pixel spacing from the DICOM metadata (assuming it's in mm)
row_spacing, col_spacing = float(0.336), float(0.336)

# Convert the image to mm by scaling the pixel array
mm_array = np.array(img_array, dtype=float)

# Convert pixel distances to mm
mm_array *= row_spacing  # Scale rows by row spacing
mm_array = mm_array.T  # Transpose to scale columns
mm_array *= col_spacing  # Scale columns by col spacing

# Now mm_array contains the image in millimeter units
print(f"mm_array shape: {mm_array.shape}")  # Check the dimensions of the mm_array

# Define the radiation field and light field positions (in mm)
light_field_positions = {
    "left": 75.48,  # Light field position in mm for left
    "right": 74.14,  # Light field position in mm for right
    "top": 75.48,    # Light field position in mm for top
    "bottom": 73.24  # Light field position in mm for bottom
}

radiation_field_positions = {
    "left": 77.20,  # Radiation field position in mm for left
    "right": 77.20,  # Radiation field position in mm for right
    "top": 77.20,    # Radiation field position in mm for top
    "bottom": 77.20  # Radiation field position in mm for bottom
}

# Central Crosshair Position (in mm)
crosshair_center_x = 74.81  # X position of crosshair center (mm)
crosshair_center_y = 74.36  # Y position of crosshair center (mm)

# BB data (in mm)
bb_offsets = [
    (15.23, 58.24, 7.62),
    (-15.01, 58.24, 7.39),
    (59.14, 14.34, 7.39),
    (59.14, -15.68, 7.17),
    (-60.93, -15.68, 7.17),
    (14.78, -60.03, 7.17),
    (-60.03, 14.34, 7.17),
    (-14.56, -60.93, 7.62)
]

# Convert positions and BB offsets to pixel space (assuming 1mm = 1px for simplicity)
mm_to_pixel = 1 / 0.336  # Inverse of the scaling factor to convert from mm to pixels

# Convert light field and radiation field positions from mm to pixels
light_field_positions_pixel = {
    key: value * mm_to_pixel for key, value in light_field_positions.items()
}

radiation_field_positions_pixel = {
    key: value * mm_to_pixel for key, value in radiation_field_positions.items()
}

# Convert BB positions and radii from mm to pixels
bb_offsets_pixel = [(x * mm_to_pixel, y * mm_to_pixel, r * mm_to_pixel) for x, y, r in bb_offsets]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))

# Calculate the center of the image (in mm)
center_x_mm = mm_array.shape[1] * col_spacing / 2
center_y_mm = mm_array.shape[0] * row_spacing / 2

# Display the DICOM image (convert to mm scale)
# Adjust the extent to center the image at (0,0)
ax.imshow(mm_array, cmap="gray", origin="lower", extent=[-center_x_mm, center_x_mm, -center_y_mm, center_y_mm])

# Plot the radiation field and light field positions
ax.plot([light_field_positions_pixel["left"], light_field_positions_pixel["right"]],
        [light_field_positions_pixel["top"], light_field_positions_pixel["bottom"]],
        "ro", markersize=10, label="Light Field Positions")

ax.plot([radiation_field_positions_pixel["left"], radiation_field_positions_pixel["right"]],
        [radiation_field_positions_pixel["top"], radiation_field_positions_pixel["bottom"]],
        "go", markersize=10, label="Radiation Field Positions")

# Plot the BB locations as circles
for x, y, r in bb_offsets_pixel:
    circle = plt.Circle((x, y), r, color="b", fill=False, linestyle="--", linewidth=2)
    ax.add_artist(circle)

# Plot the Crosshair center
ax.plot(crosshair_center_x * mm_to_pixel, crosshair_center_y * mm_to_pixel, "kx", markersize=12, label="Crosshair Center")

# Labeling the fields and adding a legend
ax.text(light_field_positions_pixel["left"], light_field_positions_pixel["bottom"], "Light Field", color="red", fontsize=12)
ax.text(radiation_field_positions_pixel["left"], radiation_field_positions_pixel["bottom"], "Radiation Field", color="green", fontsize=12)
ax.legend(loc="upper right")

# Display the plot with grid and title
ax.set_title("Overlay of Radiation and Light Field with BB Positions")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.grid(True)

# Show the plot
plt.show()


# Initialize Winston-Lutz analysis
#wl = WinstonLutzMultiTargetMultiField(wl_path)

# Perform analysis with the loaded BB arrangement
#wl.analyze(bb_arrangement=arrange)

# Print the results
#print(wl.results())