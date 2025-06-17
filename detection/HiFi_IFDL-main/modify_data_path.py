input_file = "/home/ids/saimeur-22/Projects/Gen-Ai/Detectors/HiFi_IFDL-main/data/Coverage/fake.txt"
output_file = "/home/ids/saimeur-22/Projects/Gen-Ai/Detectors/HiFi_IFDL-main/data/Coverage/fake_2.txt"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        filename = line.strip().split("/")[-1]  # Récupère uniquement le nom du fichier
        f_out.write(filename + "\n")  # Écrit le nom dans le nouveau fichier

print("Fichier modifié avec succès !")