
import csv
from pathlib import Path
from collections import Counter

import SimpleITK as sitk

root = Path.home() / "Downloads" / "picai" / "picai"
marksheet = Path.home() / "Downloads" / "marksheet.csv"

with open(marksheet, newline="", encoding="utf-8-sig") as f:
    cases = list(csv.DictReader(f))

modalities = ["t2.mha", "adc.mha", "dwi.mha"]

for modality in modalities:

    shapes = Counter()
    spacings = Counter()
    slice_counts = []
    errors = []

    for case in cases:
        path = (
            root
            / case["patient_id"]
            / case["study_id"]
            / modality
        )

        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))

            # Read metadata without loading the full MRI
            reader.ReadImageInformation()

            size = tuple(reader.GetSize())
            spacing = tuple(round(v, 2) for v in reader.GetSpacing())

            shapes[size] += 1
            spacings[spacing] += 1
            slice_counts.append(size[2])

        except Exception as error:
            errors.append((str(path), str(error)))

    print(f"\n===== {modality} =====")
    print("Successfully inspected:", len(slice_counts))

    if slice_counts:
        print("Minimum number of slices:", min(slice_counts))
        print("Maximum number of slices:", max(slice_counts))

    print("\nMost common image sizes (x, y, z):")
    for size, count in shapes.most_common(5):
        print(size, ":", count)

    print("\nMost common voxel spacings (mm):")
    for spacing, count in spacings.most_common(5):
        print(spacing, ":", count)

    print("Files with reading errors:", len(errors))

    for path, error in errors[:3]:
        print("Example error:", path, error)
