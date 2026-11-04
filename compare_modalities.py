
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt

# MRI examination
folder = (
    Path.home()
    / "Downloads"
    / "picai"
    / "picai"
    / "10131"
    / "1000132"
)

modalities = {
    "T2": "t2.mha",
    "ADC": "adc.mha",
    "DWI": "dwi.mha"
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (name, filename) in zip(
    axes, modalities.items()
):

    # Read MRI
    image = sitk.ReadImage(str(folder / filename))

    # Convert to NumPy array
    volume = sitk.GetArrayFromImage(image)

    # Print image properties
    print(f"\n{name}")
    print("Dimensions:", volume.shape)
    print("Voxel spacing:", image.GetSpacing())
    print("Origin:", image.GetOrigin())

    # Select central slice
    middle = volume.shape[0] // 2
    slice_2d = volume[middle]

    # Display
    ax.imshow(slice_2d, cmap="gray")
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.show()
