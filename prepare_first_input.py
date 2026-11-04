
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

folder = (
    Path.home() / "Downloads" / "picai"
    / "picai" / "10005" / "1000005"
)

# Load MRI volumes
t2 = sitk.ReadImage(str(folder / "t2.mha"))
adc = sitk.ReadImage(str(folder / "adc.mha"))
dwi = sitk.ReadImage(str(folder / "dwi.mha"))

# Resample ADC and DWI to the T2 grid
def align_to_t2(image):
    return sitk.Resample(
        image, t2,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )

adc = align_to_t2(adc)
dwi = align_to_t2(dwi)

# Convert to NumPy
volumes = [
    sitk.GetArrayFromImage(image)
    for image in [t2, adc, dwi]
]

# Select the central slice
z = volumes[0].shape[0] // 2
slices = [volume[z] for volume in volumes]

# Centre crop: 224 x 224 pixels
def centre_crop(image, size=224):
    h, w = image.shape
    assert h >= size and w >= size
    y = (h - size) // 2
    x = (w - size) // 2
    return image[y:y+size, x:x+size]

slices = [centre_crop(s) for s in slices]

# Simple scaling for visualisation only
def normalise_for_preview(image):
    image = image.astype(np.float32)
    mask = image > 0
    if not np.any(mask):
        return np.zeros_like(image)

    low, high = np.percentile(image[mask], [1, 99])
    result = np.clip(
        (image - low) / max(high - low, 1e-6),
        0, 1
    )
    result[~mask] = 0
    return result

slices = [normalise_for_preview(s) for s in slices]

# Combine into a three-channel array
input_3ch = np.stack(slices, axis=0)

print("Input shape:", input_3ch.shape)

# Display each channel
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

for ax, image, name in zip(
    axes, slices, ["T2", "ADC", "DWI"]
):
    ax.imshow(image, cmap="gray")
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.show()

import torch
from torch import nn
from torchvision.models import efficientnet_b0

# 1. Convert our MRI array into a PyTorch tensor
x = torch.from_numpy(input_3ch).float()

# 2. Add a batch dimension
x = x.unsqueeze(0)

print("EfficientNet input:", x.shape)

# 3. Create EfficientNet-B0 without pretrained weights
model = efficientnet_b0(weights=None)

# 4. Replace the classification head with 3 outputs
n_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(n_features, 3)

# 5. Run a forward pass
model.eval()

with torch.no_grad():
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)

print("Output shape:", logits.shape)
print("Class scores:", logits)
print("Probabilities:", probabilities)

import csv

# Read the saved patient-level split
with open("patient_splits.csv", newline="") as f:
    records = list(csv.DictReader(f))

# Find the examination used in our first EfficientNet test
case = next(
    row for row in records
    if row["patient_id"] == "10005"
    and row["study_id"] == "1000005"
)

print("\nCase information:")
print("Patient ID:", case["patient_id"])
print("Study ID:", case["study_id"])
print("True ISUP grade:", case["case_ISUP"])
print("Model class:", case["model_class"])
print("Dataset split:", case["split"])

# ==========================================
# FIRST TRAINING STEP - EFFICIENTNET
# ==========================================

# Safety check: use training data only
assert case["split"] == "train"

# True label: Class 2
target = torch.tensor(
    [int(case["model_class"])],
    dtype=torch.long
)

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer: update all EfficientNet parameters
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# Enable training mode
model.train()

# Save the weights of the first convolution
weights_before = model.features[0][0].weight.detach().clone()

# 1. Clear old gradients
optimizer.zero_grad()

# 2. Forward pass
logits = model(x)

# 3. Calculate the error
loss = loss_fn(logits, target)

print("\nFIRST TRAINING STEP")
print("True class:", target.item())
print("Loss:", loss.item())

# 4. Backpropagation
loss.backward()

gradient = model.features[0][0].weight.grad
print("Gradient norm:", gradient.norm().item())

# 5. Update the weights
optimizer.step()

# Check that weights actually changed
weights_after = model.features[0][0].weight.detach()

change = (weights_after - weights_before).abs().max().item()

print("Maximum weight change:", change)
print("Weights updated:", change > 0)
