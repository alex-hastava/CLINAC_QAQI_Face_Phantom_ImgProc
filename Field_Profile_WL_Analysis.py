import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylinac import FieldProfileAnalysis, Centering, Normalization, Edge
from pylinac.metrics.profile import (
    PenumbraLeftMetric,
    PenumbraRightMetric,
    SymmetryAreaMetric,
    FlatnessDifferenceMetric,
)
from pylinac import WinstonLutzMultiTargetMultiField
from collections import namedtuple

# Rad_Onc_PC
dicom_path = "C:/Users/ahastava/PycharmProjects/Face_Phantom_MeV_Scan.dcm"
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

#########################################################################################
# Perform Winston-Lutz analysis

# Define a named tuple for BB configuration
BB = namedtuple("BB", ["offset_left_mm", "offset_up_mm", "offset_in_mm", "bb_size_mm", "rad_size_mm"])

# Load BB configuration data
file_path = "my_special_phantom_bbs.npy"
loaded_bbs = np.load(file_path, allow_pickle=True)

# Convert to a list of named tuples
my_special_phantom_bbs = [BB(*bb) for bb in loaded_bbs]

# Initialize the Winston-Lutz MultiTargetMultiField analysis object
wl = WinstonLutzMultiTargetMultiField(wl_path)

# Pass the list of dictionaries directly to the Winston-Lutz analysis
wl.analyze(bb_arrangement=my_special_phantom_bbs)

# Plot all the analyzed Winston-Lutz images
wl.plot_images()