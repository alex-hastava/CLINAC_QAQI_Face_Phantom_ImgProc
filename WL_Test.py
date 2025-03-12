from pylinac.winston_lutz import BBArrangement

import pylinac
from pylinac.core.image_generator import simulators, layers, generate_winstonlutz_multi_bb_multi_field
from collections import namedtuple

# Define a BB class that inherits from namedtuple and includes a custom to_human method
class BB(namedtuple("BB", ["name", "offset_left_mm", "offset_up_mm", "offset_in_mm", "bb_size_mm", "rad_size_mm"])):
    def to_human(self):
        # Format description as 'BB Name: Left Xmm, Up Ymm, In Zmm'
        return f"'{self.name}': Left {self.offset_left_mm}mm, Up {self.offset_up_mm}mm, In {self.offset_in_mm}mm"

# Directory where the Winston-Lutz images will be saved
wl_dir = 'wl_dir'

# Generate Winston-Lutz multi-field with multi-BB configuration
generate_winstonlutz_multi_bb_multi_field(
    simulator=simulators.AS1200Image(sid=1000),
    field_layer=layers.PerfectFieldLayer,
    final_layers=[layers.GaussianFilterLayer(sigma_mm=1),],
    dir_out=wl_dir,
    field_offsets=((0, 0, 0), (20, -20, 60)),
    field_size_mm=(20, 20),
    bb_offsets=[[0, 0, 0], [20, -20, 60]],
)

# Create the BB arrangement as a list of namedtuples
arrange = [
    BB(name='Iso', offset_left_mm=0, offset_up_mm=0, offset_in_mm=0, bb_size_mm=5, rad_size_mm=20),
    BB(name='Left,Down,In', offset_left_mm=20, offset_up_mm=-20, offset_in_mm=60, bb_size_mm=5, rad_size_mm=20)
]

# Initialize Winston-Lutz analysis
wl = pylinac.WinstonLutzMultiTargetMultiField(wl_dir)

# Perform analysis with the correct BB arrangement
wl.analyze(bb_arrangement=arrange)

# Print the results
print(wl.results())

# plot all the images
wl.plot_images()
