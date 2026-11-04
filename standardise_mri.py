
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

root = Path.home() / "Downloads" / "picai" / "picai"

# Two examples: one regular and one with differing grids
cases = [
    ("10005", "1000005"),
    ("10009", "1000009")
]

def prepare_case(patient, study):
    folder = root / patient / study

    t2 = sitk.ReadImage(str(folder / "t2.mha"))
    adc = sitk.ReadImage(str(folder / "adc.mha"))
    dwi = sitk.ReadImage(str(folder / "dwi.mha"))

    # Common physical grid: 224 x 224 at 0.8 mm
    reference = sitk.Image(
        [224, 224, t2.GetSize()[2]],
        sitk.sitkFloat32
    )

    reference.SetSpacing(
        (0.8, 0.8, t2.GetSpacing()[2])
    )
    reference.SetDirection(t2.GetDirection())

    # Locate the physical centre of the T2 volume
    t2_centre = t2.TransformContinuousIndexToPhysicalPoint(
        [(n - 1) / 2 for n in t2.GetSize()]
    )

    ref_centre = reference.TransformContinuousIndexToPhysicalPoint(
        [(n - 1) / 2 for n in reference.GetSize()]
    )

    reference.SetOrigin(
        tuple(a - b for a, b in zip(t2_centre, ref_centre))
    )

    # Resample all modalities independently
    arrays = []

    for image in [t2, adc, dwi]:
        resampled = sitk.Resample(
            image,
            reference,
            sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32
        )

        array = sitk.GetArrayFromImage(resampled)
        arrays.append(array)

    # Select a central slice for preview only
    z = reference.GetSize()[2] // 2

    channels = []

    for array in arrays:
        image = np.nan_to_num(
            array[z].astype(np.float32)
        )

        positive = image[image > 0]

        if len(positive):
            low, high = np.percentile(positive, [1, 99])
            image = np.clip(
                (image - low) / max(high - low, 1e-6),
                0, 1
            )
        else:
            image = np.zeros_like(image)

        channels.append(image)

    return np.stack(channels, axis=0)


fig, axes = plt.subplots(2, 3, figsize=(11, 7))


for row, (patient, study) in enumerate(cases):

    x = prepare_case(patient, study)

    print(f"\nPatient {patient}")
    print("Input shape:", x.shape)
    print("All values finite:", np.isfinite(x).all())
    print("Minimum intensity:", x.min())
    print("Maximum intensity:", x.max())

    for channel, name in enumerate(["T2", "ADC", "DWI"]):
        nonzero = np.count_nonzero(x[channel])
        print(f"{name} nonzero pixels:", nonzero)

    for col, name in enumerate(["T2", "ADC", "DWI"]):
        axes[row, col].imshow(x[col], cmap="gray")
        axes[row, col].set_title(f"{patient} - {name}")
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()

# =====================================
# RESEARCH TASK 21 - PYTORCH DATASET
# =====================================

import csv
import torch
from torch.utils.data import Dataset, DataLoader


class PICAIDataset(Dataset):

    def __init__(self, split):

        with open("patient_splits.csv", newline="") as f:
            records = list(csv.DictReader(f))

        self.records = [
            row for row in records
            if row["split"] == split
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        patient = record["patient_id"]
        study = record["study_id"]

        # Use the function we created earlier
        image = prepare_case(patient, study)

        # Convert the MRI to a PyTorch tensor
        x = torch.from_numpy(
            image.astype(np.float32)
        )

        # Get the correct ISUP class
        y = torch.tensor(
            int(record["model_class"]),
            dtype=torch.long
        )

        return x, y


# Create our preliminary training and validation datasets
train_dataset = PICAIDataset("train")
val_dataset = PICAIDataset("validation")

print("\nPYTORCH DATASET")
print("Training examinations:", len(train_dataset))
print("Validation examinations:", len(val_dataset))

# Create a DataLoader with batches of 2 examinations
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

# Load the first batch
images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch labels:", labels)
print("Image dtype:", images.dtype)
