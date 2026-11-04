
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Locate one MRI from patient 10131
mri_path = (
    Path.home()
    / "Downloads"
    / "picai"
    / "picai"
    / "10131"
    / "1000132"
    / "t2.mha"
)

# Read the MRI
image = sitk.ReadImage(str(mri_path))

# Convert to a numerical array
volume = sitk.GetArrayFromImage(image)

# Examine the MRI properties
print("MRI dimensions:", volume.shape)
print("Voxel spacing (mm):", image.GetSpacing())
print("Minimum intensity:", volume.min())
print("Maximum intensity:", volume.max())

# Select the central slice of the volume
middle_slice = volume.shape[0] // 2

# Display the image
plt.imshow(volume[middle_slice], cmap="gray")
plt.title("T2-weighted prostate MRI")
plt.axis("off")
plt.show()
