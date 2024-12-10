"""DataLoader class to read and store the data"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import Audio


class DataLoader:
    def __init__(self, path_x: str, path_y: str, verbose: bool = True):
        self.path_x = path_x
        self.path_y = path_y

        self.data_x, self.freq_x = self.read_wav_files(self.path_x)
        self.data_y, self.freq_y = self.read_wav_files(self.path_y)
        self.verbose = verbose
        self.import_summary()

    def import_summary(self):
        """
        Briefly describe the import status
        """
        print("Import summary:")
        print("- Data imported successfully!")
        print(f"- Number of noised samples: {len(self.data_x)}")
        print(f"- Number of original samples: {len(self.data_y)}")
        print(
            f"- Signal shapes in noised samples: {np.unique([file.shape[0] for file in self.data_x.values()])}"
        )
        print(
            f"- Signal shapes in original samples: {np.unique([file.shape[0] for file in self.data_y.values()])}"
        )
        print(
            f"- Correspondance between both data sets: {set(self.data_x.keys()) == set(self.data_y.keys())}."
        )

    def read_wav_files(self, path: str):
        """
        Parcourt un répertoire pour lire tous les fichiers .wav et retourne un dictionnaire avec
        les noms de fichiers (sans extension) comme clés et leurs contenus (array) comme valeurs.

        Args:
            path (str): Chemin vers le dossier contenant les fichiers .wav.

        Returns:
            dict: Un dictionnaire {nom: array}.
        """
        wav_dict = {}
        frequences = {}

        for file_name in os.listdir(path):
            if file_name.endswith(".wav"):
                full_path = os.path.join(path, file_name)
                try:
                    freq, data = wavfile.read(full_path)
                    file_key = os.path.splitext(file_name)[0]
                    wav_dict[file_key] = data
                    frequences[file_key] = freq
                except Exception as e:
                    print(f"Erreur lors de la lecture de {file_name}: {e}")

        return wav_dict, frequences

    # def play_wav(self, file: str, noised=False):
    #     """
    #     Lit un fichier .wav et permet de l'écouter dans un notebook Jupyter.

    #     Args:
    #         file_path (str): Chemin vers le fichier .wav.

    #     Returns:
    #         Audio: Un objet audio prêt à être lu dans Jupyter.
    #     """
    #     if noised:
    #         freq, data = self.freq_x[file], self.data_x[file]
    #     else:
    #         freq, data = self.freq_y[file], self.data_y[file]

    #     print(f"Sampling frequency: {freq} Hz")
    #     print(f"Dimension: {data.shape}")
    #     return Audio(data, rate=freq)

    # def display_signal(self, file: str, noised=False, figsize=(10, 4)):
    #     """
    #     Lit un fichier .wav et permet de l'écouter dans un notebook Jupyter.

    #     Args:
    #         file_path (str): Chemin vers le fichier .wav.

    #     Returns:
    #         Audio: Un objet audio prêt à être lu dans Jupyter.
    #     """
    #     if noised:
    #         freq, data = self.freq_x[file], self.data_x[file]
    #     else:
    #         freq, data = self.freq_y[file], self.data_y[file]

    #     n_samples = len(data)
    #     time = np.linspace(0, n_samples / freq, n_samples)
    #     post_fix_title = "Noisy" if noised else "Original"
    #     # Affichage
    #     plt.figure(figsize=figsize)
    #     plt.plot(time, data)
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Amplitude")
    #     plt.title(f"{post_fix_title} Signal {file}")
    #     plt.grid()
    #     plt.show()
