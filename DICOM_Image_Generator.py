from pylinac.core.image_generator.utils import generate_winstonlutz_multi_bb_multi_field
from pylinac.core.image_generator.simulators import Simulator
from pylinac.core import image_generator
import json
import numpy as np

# Define the AS5000 class which inherits from Simulator
class AS5000(Simulator):
    pixel_size = 0.336  # Pixel size for the simulator
    shape = (1280, 1280)  # Shape of the simulator's image (1280x1280)

# Instantiate the simulator & base layer (AS5000 in this case)
simulator = AS5000()
field_layer = image_generator.layers.FilteredFieldLayer

# Output directory to save the images
dir_out = "C:/Users/ahastava/PycharmProjects/CLINAC_QAQI_Face_Phantom_ImgProc/wl_images"

# Define the field offsets, e.g., shifting the field by some amount
field_offsets = [[0, 0, 0], [0, 0, 0]]

circles = np.load("bb_pixel_coords.npy")

# Define BB offsets: using the ball bearing pixel data converted to mm
# Convert to a list of tuples (x_pixel, y_pixel, diameter_mm)
circle_data = [(int(x), int(y), 2 * r) for x, y, r in circles]

print("Loaded Circle Data:", circle_data)

# Pixel spacing in mm/pixel
pixel_spacing = 0.336  # Example pixel spacing (mm per pixel)

# Define the DICOM RT Image Position (assuming it corresponds to isocenter)
rt_image_position = (2/3 * (0.336 * (1280/2)), 0, 2/3 * (0.336 * (1280/2)))  # Isocenter position, (x, y, z) in mm

# Convert pixel positions to mm and calculate the 3D offsets for the BBs
bb_offsets = []
for x, z, diameter in circle_data:
    # Convert center coordinates from pixels to mm
    x_mm = 2/3 * (x * pixel_spacing)
    y_mm = 0
    z_mm = 2/3 * (z * pixel_spacing)

    # Calculate the 3D offsets from the isocenter
    x_offset = x_mm - rt_image_position[0]
    y_offset = y_mm - rt_image_position[1]
    z_offset = z_mm - rt_image_position[2]

    # The diameter is already in mm, so we use it directly as the BB size
    bb_size_mm = diameter

    # Append the BB offset and the BB size to the list
    bb_offsets.append({
        "offset_left_mm": x_offset,  # X-axis offset (left-right)
        "offset_up_mm": y_offset,    # Y-axis offset (up-down)
        "offset_in_mm": z_offset,    # Z-axis offset (in-out, assuming 0 here)
        "bb_size_mm": bb_size_mm,    # Diameter in mm
        "rad_size_mm": 13.5,         # Radiation radius surrounding ROI in mm
    })

# Convert list of dicts to structured array
dtype = np.dtype([
    ("offset_left_mm", np.float32),
    ("offset_up_mm", np.float32),
    ("offset_in_mm", np.float32),
    ("bb_size_mm", np.float32),
    ("rad_size_mm", np.float32)
])

# Convert list of dictionaries into a structured array
bb_array = np.array(
    [(bb["offset_left_mm"], bb["offset_up_mm"], bb["offset_in_mm"], bb["bb_size_mm"], bb["rad_size_mm"]) for bb in bb_offsets],
    dtype=dtype
)

# Save structured array as .npy
np.save("my_special_phantom_bbs.npy", bb_array)

print("Saved my_special_phantom_bbs.npy successfully!")

# Now bb_offsets contains the 3D offsets of the ball bearings in mm, relative to the isocenter.
# print(bb_offsets)

# Define field size in mm (e.g., based on your DICOM metadata, this might vary)
field_size_mm = (147, 147)  # Example field size

# Define additional layers, for example, to apply blurring or noise (optional)
final_layers = None  # No additional layers applied in this case

# Define BB size in mm (using a typical BB size, but adjust as needed)
bb_size_mm = 15

# Image axes: define multiple gantry angles for simulation
image_axes = [
    (0, 0, 0),              # Gantry angle 0
    (90, 0, 0),             # Gantry angle 90
    (180, 0, 0),            # Gantry angle 180
    (270, 0, 0)             # Gantry angle 270
]

# Gantry tilt and sag to simulate effects (from your metadata, slight tilt or sag)
gantry_tilt = 0.00663411039557  # Based on the metadata provided
gantry_sag = 0.0096325257373   # Simulate some sag

# Optional jitter in mm for randomness in BB positioning (if desired)
jitter_mm = 0.1  # Example jitter in mm

# Clean directory flag (whether to clean output directory before generating new images)
clean_dir = True

# Alignment to pixels (ensure the BB is aligned correctly with pixels)
align_to_pixels = True

# Call the function to generate images
generated_images = generate_winstonlutz_multi_bb_multi_field(
    simulator=simulator,  # The image simulator
    field_layer=field_layer,  # The primary field layer simulating radiation
    dir_out=dir_out,  # The directory to save the images to
    field_offsets=field_offsets,  # A list of lists containing the shift of the fields
    bb_offsets=bb_offsets,  # A list of lists containing the shift of the BBs from iso
    field_size_mm=field_size_mm,  # The field size of the radiation field in mm
    final_layers=final_layers,  # Layers to apply after generating the primary field and BB layer
    bb_size_mm=bb_size_mm,  # The size of the BB
    image_axes=image_axes,  # List of axis values for the images
    gantry_tilt=gantry_tilt,  # The tilt of the gantry in degrees
    gantry_sag=gantry_sag,  # The sag of the gantry
    clean_dir=clean_dir,  # Whether to clean out the output directory
    jitter_mm=jitter_mm  # The amount of jitter to add to the BB location in mm
)

# Output the generated image paths
print(generated_images)
