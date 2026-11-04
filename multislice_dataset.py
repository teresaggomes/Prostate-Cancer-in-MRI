
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader

root = Path.home() / "Downloads" / "picai" / "picai"


def prepare_multislice(patient, study):
    folder = root / patient / study

    t2 = sitk.ReadImage(str(folder / "t2.mha"))
    adc = sitk.ReadImage(str(folder / "adc.mha"))
    dwi = sitk.ReadImage(str(folder / "dwi.mha"))

    # Create a common physical grid
    reference = sitk.Image(
        [224, 224, t2.GetSize()[2]],
        sitk.sitkFloat32
    )

    reference.SetSpacing(
        (0.8, 0.8, t2.GetSpacing()[2])
    )
    reference.SetDirection(t2.GetDirection())

    centre = t2.TransformContinuousIndexToPhysicalPoint(
        [(n - 1) / 2 for n in t2.GetSize()]
    )

    offset = reference.TransformContinuousIndexToPhysicalPoint(
        [(n - 1) / 2 for n in reference.GetSize()]
    )

    reference.SetOrigin(
        tuple(a - b for a, b in zip(centre, offset))
    )

    # Resample T2, ADC and DWI
    volumes = []

    for image in [t2, adc, dwi]:
        aligned = sitk.Resample(
            image, reference,
            sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkLinear, 0.0, sitk.sitkFloat32
        )
        volumes.append(sitk.GetArrayFromImage(aligned))

    # Select three positions within the volume
    n_slices = volumes[0].shape[0]

    indices = [
        round((n_slices - 1) * fraction)
        for fraction in [0.35, 0.50, 0.65]
    ]

    exam_slices = []

    for z in indices:
        channels = []

        for volume in volumes:
            image = np.nan_to_num(
                volume[z].astype(np.float32)
            )

            mask = image > 0

            if mask.any():
                low, high = np.percentile(
                    image[mask], [1, 99]
                )
                image = np.where(
                    mask,
                    np.clip(
                        (image - low) / max(high - low, 1e-6),
                        0, 1
                    ),
                    0
                )
            else:
                image = np.zeros_like(image)

            channels.append(image.astype(np.float32))

        exam_slices.append(np.stack(channels))

    result = np.stack(exam_slices)

    return torch.from_numpy(result).float()


class PICAIMultiSliceDataset(Dataset):

    def __init__(self, split):
        with open("patient_splits.csv", newline="") as f:
            records = list(csv.DictReader(f))

        self.records = [
            r for r in records if r["split"] == split
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image = prepare_multislice(
            record["patient_id"],
            record["study_id"]
        )

        label = int(record["model_class"])

        return image, torch.tensor(
            label, dtype=torch.long
        )


train_dataset = PICAIMultiSliceDataset("train")
val_dataset = PICAIMultiSliceDataset("validation")

loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

images, labels = next(iter(loader))

print("Training examinations:", len(train_dataset))
print("Validation examinations:", len(val_dataset))
print("Batch shape:", images.shape)
print("Batch labels:", labels)
print("All finite:", torch.isfinite(images).all().item())
