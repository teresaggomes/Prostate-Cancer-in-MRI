
import csv
from collections import Counter

# Load our preliminary split
with open("patient_splits.csv", newline="") as f:
    records = list(csv.DictReader(f))

# Use validation data only
validation = [
    row for row in records
    if row["split"] == "validation"
]

# True labels
y_true = [
    int(row["model_class"])
    for row in validation
]

# Simple model: always predict Class 0
y_pred = [0] * len(y_true)

# Calculate accuracy
correct = sum(
    true == pred
    for true, pred in zip(y_true, y_pred)
)

accuracy = correct / len(y_true)

# Calculate F1-score for each class
f1_scores = []

for klass in range(3):
    tp = sum(
        t == klass and p == klass
        for t, p in zip(y_true, y_pred)
    )
    fp = sum(
        t != klass and p == klass
        for t, p in zip(y_true, y_pred)
    )
    fn = sum(
        t == klass and p != klass
        for t, p in zip(y_true, y_pred)
    )

    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 0
    f1_scores.append(f1)

macro_f1 = sum(f1_scores) / 3

print("MAJORITY-CLASS BASELINE")
print("Validation examinations:", len(y_true))
print("Class distribution:", dict(Counter(y_true)))
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

for klass, f1 in enumerate(f1_scores):
    print(f"Class {klass} F1: {f1:.4f}")
