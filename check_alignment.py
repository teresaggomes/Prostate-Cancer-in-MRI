
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt

folder = (
    Path.home() / "Downloads" / "picai"
    / "picai" / "10005" / "1000005")

# Load the original images
t2 = sitk.ReadImage(str(folder / "t2.mha"))
adc = sitk.ReadImage(str(folder / "adc.mha"))

# Resample ADC to the T2 physical grid
adc_aligned = sitk.Resample(
    adc,
    t2,
    sitk.Transform(3, sitk.sitkIdentity),
    sitk.sitkLinear,
    0.0,
    sitk.sitkFloat32
)

# Convert images to NumPy arrays
t2_array = sitk.GetArrayFromImage(t2)
adc_array = sitk.GetArrayFromImage(adc_aligned)

# Choose the same slice index
middle = t2_array.shape[0] // 2

t2_slice = t2_array[middle]
adc_slice = adc_array[middle]


# Select three different MRI slices
slices = [4, 9, 14]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, z in enumerate(slices):

    # First row: original T2 slices
    axes[0, i].imshow(t2_array[z], cmap="gray")
    axes[0, i].set_title(f"T2 - Slice {z}")
    axes[0, i].axis("off")

    # Second row: T2 + resampled ADC
    axes[1, i].imshow(t2_array[z], cmap="gray")
    axes[1, i].imshow(
        adc_array[z],
        cmap="magma",
        alpha=0.30
    )
    axes[1, i].set_title(f"Overlay - Slice {z}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()
