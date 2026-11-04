
import csv
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import efficientnet_b0

# Reproducibility and CPU settings
torch.manual_seed(42)
torch.set_num_threads(4)

root = Path.home() / "Downloads" / "picai" / "picai"

# Load our preliminary patient split
with open("patient_splits.csv", newline="") as f:
    records = list(csv.DictReader(f))

train_cases = [
    r for r in records if r["split"] == "train"
]

# Select two different patients per class
rng = random.Random(42)
selected = []
used_patients = set()

for klass in range(3):
    candidates = [
        r for r in train_cases
        if int(r["model_class"]) == klass
    ]
    rng.shuffle(candidates)

    count = 0
    for r in candidates:
        if r["patient_id"] not in used_patients:
            selected.append(r)
            used_patients.add(r["patient_id"])
            count += 1
        if count == 2:
            break

    assert count == 2

# Load and prepare three slices per examination
def prepare_case(record):
    folder = (
        root / record["patient_id"] / record["study_id"]
    )

    t2 = sitk.ReadImage(str(folder / "t2.mha"))
    adc = sitk.ReadImage(str(folder / "adc.mha"))
    dwi = sitk.ReadImage(str(folder / "dwi.mha"))

    transform = sitk.Transform(3, sitk.sitkIdentity)

    adc = sitk.Resample(
        adc, t2, transform,
        sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    dwi = sitk.Resample(
        dwi, t2, transform,
        sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )

    volumes = [
        sitk.GetArrayFromImage(im)
        for im in [t2, adc, dwi]
    ]

    z = volumes[0].shape[0] // 2
    indices = [max(0, z-2), z, min(2*z, z+2)]

    slices = []

    for index in indices:
        channels = []

        for volume in volumes:
            image = volume[index].astype(np.float32)
            image = np.nan_to_num(image)

            h, w = image.shape
            size = min(224, h, w)
            y0 = (h-size) // 2
            x0 = (w-size) // 2
            image = image[y0:y0+size, x0:x0+size]

            positive = image[image > 0]

            if len(positive):
                low, high = np.percentile(
                    positive, [1, 99]
                )
                image = np.clip(
                    (image-low)/max(high-low, 1e-6),
                    0, 1
                )
            else:
                image = np.zeros_like(image)

            tensor = torch.from_numpy(
                image.astype(np.float32)
            )[None, None]

            tensor = F.interpolate(
                tensor, size=(128, 128),
                mode="bilinear",
                align_corners=False
            )

            channels.append(tensor[0, 0])

        slices.append(torch.stack(channels))

    return torch.stack(slices)

# Create EfficientNet from scratch
model = efficientnet_b0(weights=None)

features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(features, 3)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0001
)

model.train()

print("TRAINING SMOKE TEST")
print("Examinations:", len(selected))

for i, record in enumerate(selected, start=1):
    images = prepare_case(record)
    label = torch.tensor(
        [int(record["model_class"])],
        dtype=torch.long
    )

    optimizer.zero_grad()

    # Three slice predictions -> one examination score
    slice_logits = model(images)
    case_logits = slice_logits.mean(dim=0, keepdim=True)

    loss = loss_fn(case_logits, label)
    loss.backward()
    optimizer.step()

    print(
        f"Case {i}/6 | "
        f"Class {label.item()} | "
        f"Loss {loss.item():.4f}",
        flush=True
    )

print("Small training test completed!")
