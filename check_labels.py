
import csv
from pathlib import Path
from collections import Counter

# Load the PI-CAI marksheet
marksheet = Path.home() / "Downloads" / "marksheet.csv"

with open(marksheet, newline="", encoding="utf-8-sig") as f:
    cases = list(csv.DictReader(f))

# Check the available columns
print("Available columns:")
print(list(cases[0].keys()))

# Focus on the ISUP 0 examinations
isup_0 = [
    case for case in cases
    if case["case_ISUP"].strip() == "0"
]

print("\nTotal ISUP 0 examinations:", len(isup_0))

# Investigate recorded histopathology types
histopath_types = Counter(
    case["histopath_type"].strip() or "Not recorded"
    for case in isup_0
)

print("\nHistopathology information for ISUP 0:")
for type_name, count in histopath_types.items():
    print(type_name, ":", count)

# Check whether ISUP grades and csPCa labels agree
inconsistencies = 0

for case in cases:
    grade = int(case["case_ISUP"])
    csPCa = case["case_csPCa"].strip().upper()

    expected = "YES" if grade >= 2 else "NO"

    if csPCa != expected:
        inconsistencies += 1

print("\nISUP / csPCa inconsistencies:", inconsistencies)
