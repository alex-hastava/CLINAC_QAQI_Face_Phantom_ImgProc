import numpy as np
import random
from pylinac.core import image_generator
from pylinac.core.image_generator import AS1200Image
from pylinac.core.image_generator.layers import PerfectFieldLayer, GaussianFilterLayer
from pylinac.core.image_generator import simulators, generate_winstonlutz_multi_bb_multi_field

# Define field size in mm scaled down by a factor of 1/2
FIELD_SIZE_MM = (20, 20)

# Define BB size in mm
BB_SIZE_MM = 7.5

# Output directory
DIR_OUT = r"C:/Users/ahastava/PycharmProjects/CLINAC_QAQI_Face_Phantom_ImgProc/wl_images"

# Define field offsets
FIELD_OFFSETS = [[0, 0, 0], [0, 0, 0]]

# Load BB pixel coordinates
circles = np.load("wl_bb_pixel_coords.npy")

# Pixel spacing in mm/pixel
PIXEL_SPACING = 0.336

# Define RT Image Position at isocenter
RT_IMAGE_POSITION = (2/3 * (PIXEL_SPACING * (1280/2)), 2/3 * (PIXEL_SPACING * (1280/2)), 2/3 * (PIXEL_SPACING * (1280/2)))

bb_offsets = [
    {
        "offset_left_mm": (2/3) * (x * PIXEL_SPACING) - RT_IMAGE_POSITION[0],
        "offset_up_mm": (2/3) * (((x + z) / 2) * PIXEL_SPACING - RT_IMAGE_POSITION[1]) - (index * 10),
        "offset_in_mm": (2/3) * (z * PIXEL_SPACING) - RT_IMAGE_POSITION[2]
    }
    for index, (x, z, _) in enumerate(circles, start=1)
]

# Convert list of dicts to structured array
dtype = np.dtype([
    ("offset_left_mm", np.float32),
    ("offset_up_mm", np.float32),
    ("offset_in_mm", np.float32)
])

bb_array = np.array(
    [(bb["offset_left_mm"], bb["offset_up_mm"], bb["offset_in_mm"]) for bb in bb_offsets],
    dtype=dtype
)

# Save structured array
np.save("wl_special_phantom_bbs_mm.npy", bb_array)
print("Saved wl_special_phantom_bbs_mm.npy successfully!")
print(bb_array)

# Define image acquisition angles
IMAGE_AXES = [
    (0, 0, 0),
    #(90, 0, 0),
    (180, 0, 0)#,
    #(270, 0, 0)
]

# Gantry effects
GANTRY_TILT = 0.00663411039557
GANTRY_SAG = 0.0096325257373

as1200 = AS1200Image()  # this will set the pixel size and shape automatically
field_layer = image_generator.layers.PerfectFieldLayer

# Generate WL images
generated_images = generate_winstonlutz_multi_bb_multi_field(
    simulator=simulators.AS1200Image(sid=1500.03497428306),
    field_layer=field_layer,
    final_layers=[GaussianFilterLayer(sigma_mm=0.2)],
    dir_out=DIR_OUT,
    field_offsets=tuple(tuple(bb.values()) for bb in bb_offsets),
    field_size_mm=FIELD_SIZE_MM,
    bb_offsets=bb_offsets,
    clean_dir=True
)

# Output generated images
print(generated_images)
