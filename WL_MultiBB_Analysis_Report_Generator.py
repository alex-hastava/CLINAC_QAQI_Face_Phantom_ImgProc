from pylinac.winston_lutz import BBConfig
from pylinac import WinstonLutzMultiTargetMultiField
import numpy as np
import plotly.io as pio

# Define the path to the Winston-Lutz images
#wl_path = r"C:/Users/ahastava/PycharmProjects/CLINAC_QAQI_Face_Phantom_ImgProc/wl_images"

wl_path = r"C:/Users/Hasta/PycharmProjects/CLINAC_QAQI_Face_Phantom_ImgProc/wl_images"

# Load BB coordinates from file
bb_array = np.load("wl_special_phantom_bbs_mm.npy")

# Convert my_special_phantom_bbs into a tuple of BBConfig objects
my_special_phantom_bbs = tuple(
    BBConfig(
        name=f"BB_{i}",
        offset_left_mm=bb[0],
        offset_up_mm=bb[1],
        offset_in_mm=bb[2],
        bb_size_mm=7.5,
        rad_size_mm=20
    )
    for i, bb in enumerate(bb_array)
)

for bb in my_special_phantom_bbs:
    print(bb)

# Load WL test from the given image directory
wl = WinstonLutzMultiTargetMultiField(wl_path)

wl.analyze(bb_arrangement=my_special_phantom_bbs)

# Print results
print(wl.results())

print(wl.bb_shift_instructions())

# Plot results
pio.renderers.default = 'browser'
wl.plot_images()

wl.plotly_analyzed_images()

# Publish results
wl.publish_pdf("WL_Img_Gen_Rpt.pdf")