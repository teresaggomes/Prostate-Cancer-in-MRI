
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt

folder = (
    Path.home() / "Downloads" / "picai"
    / "picai" / "10131" / "1000132"
)

# Load the three modalities
t2 = sitk.ReadImage(str(folder / "t2.mha"))
adc = sitk.ReadImage(str(folder / "adc.mha"))
dwi = sitk.ReadImage(str(folder / "dwi.mha"))

# Inspect their orientations
print("T2 direction:", t2.GetDirection())
print("ADC direction:", adc.GetDirection())
print("DWI direction:", dwi.GetDirection())

# Use the T2 grid as reference
identity = sitk.Transform(3, sitk.sitkIdentity)

adc_aligned = sitk.Resample(
    adc, t2, identity,
    sitk.sitkLinear, 0.0, sitk.sitkFloat32
)

dwi_aligned = sitk.Resample(
    dwi, t2, identity,
    sitk.sitkLinear, 0.0, sitk.sitkFloat32
)

images = {
    "T2": t2,
    "ADC resampled": adc_aligned,
    "DWI resampled": dwi_aligned
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (name, image) in zip(axes, images.items()):
    volume = sitk.GetArrayFromImage(image)

    middle = volume.shape[0] // 2

    ax.imshow(volume[middle], cmap="gray")
    ax.set_title(name)
    ax.axis("off")

    print(name, ":", volume.shape)

plt.tight_layout()
plt.show()
