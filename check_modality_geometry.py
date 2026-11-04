
import csv
from pathlib import Path
import SimpleITK as sitk

root = Path.home() / "Downloads" / "picai" / "picai"
marksheet = Path.home() / "Downloads" / "marksheet.csv"

with open(marksheet, newline="", encoding="utf-8-sig") as f:
    cases = list(csv.DictReader(f))

# Read image geometry without loading voxel data
def get_geometry(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()

    return {
        "size": reader.GetSize(),
        "spacing": reader.GetSpacing(),
        "origin": reader.GetOrigin(),
        "direction": reader.GetDirection()
    }

def close_enough(a, b, tolerance=0.001):
    return all(
        abs(x - y) < tolerance
        for x, y in zip(a, b)
    )

same_grid = 0
different_grid = []
different_slice_counts = 0

for case in cases:
    folder = (
        root / case["patient_id"] / case["study_id"]
    )

    t2 = get_geometry(folder / "t2.mha")
    adc = get_geometry(folder / "adc.mha")
    dwi = get_geometry(folder / "dwi.mha")

    if t2["size"][2] != adc["size"][2]:
        different_slice_counts += 1

    matches = (
        adc["size"] == dwi["size"]
        and close_enough(adc["spacing"], dwi["spacing"])
        and close_enough(adc["origin"], dwi["origin"])
        and close_enough(adc["direction"], dwi["direction"])
    )

    if matches:
        same_grid += 1
    else:
        different_grid.append(
            (case["patient_id"], case["study_id"])
        )

print("Total examinations:", len(cases))
print("ADC and DWI matching grids:", same_grid)
print("ADC and DWI different grids:", len(different_grid))
print("T2 and ADC different slice counts:",
      different_slice_counts)

print("\nExamples requiring inspection:")
for item in different_grid[:5]:
    print(item)
