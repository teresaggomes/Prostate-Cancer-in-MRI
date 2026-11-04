
import csv
from pathlib import Path
from collections import Counter

# Localização dos dados
downloads = Path.home() / "Downloads"
dataset = downloads / "picai" / "picai"
marksheet = downloads / "marksheet.csv"

# Ler o ficheiro com os labels
with open(marksheet, newline="", encoding="utf-8-sig") as f:
    cases = list(csv.DictReader(f))

print("Total examinations:", len(cases))

# Contar pacientes únicos
patients = {case["patient_id"] for case in cases}
print("Unique patients:", len(patients))

# Verificar as imagens de cada exame
complete = 0
incomplete = []

for case in cases:
    patient = case["patient_id"]
    study = case["study_id"]

    folder = dataset / patient / study

    required_files = [
        "t2.mha",
        "adc.mha",
        "dwi.mha"
    ]

    missing = [
        name for name in required_files
        if not (folder / name).is_file()
    ]

    if not missing:
        complete += 1
    else:
        incomplete.append((patient, study, missing))

print("Complete examinations:", complete)
print("Incomplete examinations:", len(incomplete))

# Mostrar exemplos de exames incompletos
for item in incomplete[:5]:
    print("Missing files:", item)

# Contar os graus ISUP
grades = Counter(
    case["case_ISUP"].strip()
    for case in cases
)

print("\nISUP distribution:")
for grade in range(6):
    print(f"ISUP {grade}: {grades[str(grade)]}")

print("Missing ISUP labels:", grades[""])

# Calcular as três classes do projeto
class_0 = grades["0"] + grades["1"]
class_1 = grades["2"]
class_2 = grades["3"] + grades["4"] + grades["5"]

print("\nOur three classes:")
print("Class 0:", class_0)
print("Class 1:", class_1)
print("Class 2:", class_2)
