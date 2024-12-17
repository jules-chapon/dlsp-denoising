"""DataLoader class to read and store the data"""

from dataclasses import dataclass, field
import os
import numpy as np
from scipy.io import wavfile


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
            f"- Frequencies in noised samples: {np.unique([freq for freq in self.freq_y.values()])}"
        )
        print(
            f"- Frequencies in original samples: {np.unique([freq for freq in self.freq_x.values()])}"
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

    def get_harmonized_data(self):
        """normalize and harmonize frequencies"""
        names = []
        x = []
        y = []
        sampling_freq = 4_000
        for name in list(self.data_x.keys()):
            names.append(name)
            x_downsampled = np.mean(self.data_x[name].reshape(-1, 2), axis=1)
            x.append(x_downsampled / np.max(np.abs(x_downsampled)))
            y.append(self.data_y[name] / np.max(np.abs(self.data_y[name])))

        harmonized_data = HarmonizedData(
            x=np.array(x).astype(np.float64),
            y=np.array(y).astype(np.float64),
            names=names,
            sampling_freq=sampling_freq,
        )
        return harmonized_data


@dataclass
class HarmonizedData:
    """
    Simple container for clean data
    """

    x: np.ndarray
    y: np.ndarray
    names: list[str]
    sampling_freq: float
    n_samples: int = field(
        init=False
    )  # Ce champ sera calculé, donc on utilise init=False

    def __post_init__(self):
        self.n_samples = len(self.x)
