import numpy as np
import imageio
import os

def convert_npy_to_mp4(npy_path, output_path, fps=30):
    """
    Convertit un fichier .npy en une vidéo .mp4.
    
    Parameters:
    - npy_path (str): Chemin vers le fichier .npy.
    - output_path (str): Chemin de sortie pour la vidéo .mp4.
    - fps (int): Images par seconde pour la vidéo.
    """
    # Charger le fichier .npy
    frames = np.load(npy_path)
    
    # Vérifier les dimensions du tableau
    print("Shape des frames :", frames.shape)  # Devrait être (nombre_de_frames, hauteur, largeur, canaux)

    #macro_block_size set to 1

    # Créer une vidéo à partir des frames
    with imageio.get_writer(output_path, fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            # S'assurer que les valeurs des pixels sont au bon format (par exemple, 0-255 pour uint8)
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)  # Normaliser si nécessaire
            
            writer.append_data(frame)
    
    print(f"Vidéo créée avec succès : {output_path}")

# Exemple d'utilisation de la fonction
path = os.getcwd()
npy_file_path1 = os.path.join(path,'masks.npy')
video_output_path1 = os.path.join(path,'video_masks.mp4')
npy_file_path2 = os.path.join(path,'images.npy')
video_output_path2 = os.path.join(path,'video_output.mp4')
convert_npy_to_mp4(npy_file_path1, video_output_path1, fps=8)
convert_npy_to_mp4(npy_file_path2, video_output_path2, fps=8)

