
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import efficientnet_b0

from multislice_dataset import PICAIMultiSliceDataset

torch.manual_seed(42)
torch.set_num_threads(4)

# Load existing datasets
train_data = PICAIMultiSliceDataset("train")
val_data = PICAIMultiSliceDataset("validation")


# Select 2 examinations from each class,
# ensuring different patients.
def select_examples(dataset):
    indices = []
    used_patients = set()

    for klass in range(3):
        count = 0

        for i, record in enumerate(dataset.records):
            patient = record["patient_id"]

            if (
                int(record["model_class"]) == klass
                and patient not in used_patients
            ):
                indices.append(i)
                used_patients.add(patient)
                count += 1

            if count == 2:
                break

        assert count == 2

    return Subset(dataset, indices)


train_small = select_examples(train_data)
val_small = select_examples(val_data)

train_loader = DataLoader(
    train_small, batch_size=2, shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_small, batch_size=2, shuffle=False,
    num_workers=0
)

# Create EfficientNet from scratch
model = efficientnet_b0(weights=None)

n_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(n_features, 3)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0001
)


# Aggregate slice scores into case scores
def predict_cases(images):
    batch, slices, channels, h, w = images.shape

    # Merge the batch and slice dimensions
    x = images.reshape(
        batch * slices, channels, h, w
    )

    # Smaller images for this CPU smoke test
    x = F.interpolate(
        x, size=(128, 128),
        mode="bilinear",
        align_corners=False
    )

    slice_logits = model(x)

    # Restore the slice dimension
    slice_logits = slice_logits.reshape(
        batch, slices, 3
    )

    # One prediction per examination
    return slice_logits.mean(dim=1)



# TRAINING - THREE EPOCHS

for epoch in range(1, 4):

    model.train()
    total_loss = 0.0

    for images, labels in train_loader:

        optimizer.zero_grad()

        logits = predict_cases(images)

        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    average_loss = total_loss / len(train_small)

    print(
        f"Epoch {epoch}/3 | "
        f"Training loss: {average_loss:.4f}",
        flush=True
    )



# VALIDATION
print("\nVALIDATION")
model.eval()

predictions = []
true_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        logits = predict_cases(images)

        predicted = logits.argmax(dim=1)

        predictions.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

accuracy = sum(
    p == y for p, y in zip(predictions, true_labels)
) / len(true_labels)

# Macro F1 across the three classes
f1_scores = []

for klass in range(3):
    tp = sum(
        p == klass and y == klass
        for p, y in zip(predictions, true_labels)
    )

    fp = sum(
        p == klass and y != klass
        for p, y in zip(predictions, true_labels)
    )

    fn = sum(
        p != klass and y == klass
        for p, y in zip(predictions, true_labels)
    )

    denominator = 2 * tp + fp + fn

    f1 = (
        2 * tp / denominator
        if denominator > 0 else 0
    )

    f1_scores.append(f1)

macro_f1 = sum(f1_scores) / 3

print("True labels:", true_labels)
print("Predictions:", predictions)
print("Validation accuracy:", round(accuracy, 4))
print("Validation macro F1:", round(macro_f1, 4))
print("\nTraining and validation test completed!")
