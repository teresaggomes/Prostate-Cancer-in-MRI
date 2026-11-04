
import csv
import random
from pathlib import Path
from collections import Counter

# 1. Read the marksheet
path = Path.home() / "Downloads" / "marksheet.csv"

with open(path, newline="", encoding="utf-8-sig") as f:
    cases = list(csv.DictReader(f))

# 2. Identify all unique patients
patients = sorted(set(
    case["patient_id"] for case in cases
))

# 3. Shuffle patients reproducibly
random.Random(42).shuffle(patients)

# 4. Split patients into three groups
n = len(patients)

n_train = int(0.70 * n)
n_val = int(0.15 * n)

train = set(patients[:n_train])
val = set(patients[n_train:n_train + n_val])
test = set(patients[n_train + n_val:])

# 5. Verify there is no patient overlap
assert train.isdisjoint(val)
assert train.isdisjoint(test)
assert val.isdisjoint(test)

# 6. Count examinations in each group
def report(name, patient_set):
    rows = [
        case for case in cases
        if case["patient_id"] in patient_set
    ]

    grades = Counter()

    for case in rows:
        grade = int(case["case_ISUP"])

        if grade <= 1:
            grades[0] += 1
        elif grade == 2:
            grades[1] += 1
        else:
            grades[2] += 1

    print(f"\n{name}")
    print("Patients:", len(patient_set))
    print("Examinations:", len(rows))
    print("Class 0:", grades[0])
    print("Class 1:", grades[1])
    print("Class 2:", grades[2])

report("TRAINING", train)
report("VALIDATION", val)
report("TESTING", test)

print("\nNo patient overlap: VERIFIED")

# Save the preliminary patient-level split

output = "patient_splits.csv"

with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "patient_id",
        "study_id",
        "case_ISUP",
        "model_class",
        "split"
    ])

    for case in cases:
        patient = case["patient_id"]
        grade = int(case["case_ISUP"])

        if patient in train:
            split = "train"
        elif patient in val:
            split = "validation"
        else:
            split = "test"

        if grade <= 1:
            model_class = 0
        elif grade == 2:
            model_class = 1
        else:
            model_class = 2

        writer.writerow([
            patient,
            case["study_id"],
            grade,
            model_class,
            split
        ])

print("Saved preliminary splits to:", output)
