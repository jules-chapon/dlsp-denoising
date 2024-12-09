import os
from scipy.io import wavfile
from IPython.display import Audio


class DataLoader:
    def __init__(self, path_x: str, path_y: str):
        self.path_x = path_x
        self.path_y = path_y

        self.data_x, self.freq_x = self.read_wav_files(self.path_x)
        self.data_y, self.freq_y = self.read_wav_files(self.path_y)

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

    def play_wav(self, file: str, noised=False):
        """
        Lit un fichier .wav et permet de l'écouter dans un notebook Jupyter.

        Args:
            file_path (str): Chemin vers le fichier .wav.

        Returns:
            Audio: Un objet audio prêt à être lu dans Jupyter.
        """
        if not noised:
            freq_x, data_x = self.freq_x[file], self.data_x[file]
            print(f"Fréquence d'échantillonnage : {freq_x} Hz")
            print(f"Dimensions des données : {data_x.shape}")
            return Audio(data_x, rate=freq_x)
        else:
            freq_y, data_y = self.freq_y[file], self.data_y[file]
            print(f"Fréquence d'échantillonnage : {freq_y} Hz")
            print(f"Dimensions des données : {data_y.shape}")
            return Audio(data_y, rate=freq_y)
