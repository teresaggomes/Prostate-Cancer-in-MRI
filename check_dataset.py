
from pathlib import Path

# Localizar a pasta Downloads do Windows
downloads = Path.home() / "Downloads"

# Pasta que contém as imagens MRI
dataset = downloads / "picai" / "picai"

# Ficheiro com os labels ISUP
marksheet = downloads / "marksheet.csv"

# Verificar se o Python encontra os ficheiros
print("MRI dataset found:", dataset.is_dir())
print("ISUP marksheet found:", marksheet.is_file())

# Verificar a estrutura de um paciente conhecido
study = dataset / "10131" / "1000132"

print("Study folder found:", study.is_dir())
print("T2 MRI found:", (study / "t2.mha").is_file())
print("ADC MRI found:", (study / "adc.mha").is_file())
print("DWI MRI found:", (study / "dwi.mha").is_file())
