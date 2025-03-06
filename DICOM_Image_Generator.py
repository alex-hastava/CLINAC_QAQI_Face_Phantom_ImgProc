from matplotlib import pyplot as plt

from pylinac.core.image_generator import AS1200Image
from pylinac.core.image_generator.layers import FilteredFieldLayer, GaussianFilterLayer

as1200 = AS1200Image()  # this will set the pixel size and shape automatically
as1200.add_layer(FilteredFieldLayer(field_size_mm=(50, 50)))  # create a 50x50mm square field
as1200.add_layer(GaussianFilterLayer(sigma_mm=2))  # add an image-wide gaussian to simulate penumbra/scatter
as1200.generate_dicom(file_out_name="my_AS1200.dcm", gantry_angle=45)  # create a DICOM file with the simulated image
# plot the generated image
plt.imshow(as1200.image)
plt.show()