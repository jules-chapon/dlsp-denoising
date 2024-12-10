"""Visualization functions"""

from src.libs.preprocessing import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.signal import stft


class DataVisualizer:
    """
    Permet de visualiser la forme d'onde, d'écouter un signal et d'afficher son spectrogramme.
    """

    def __init__(self, data: DataLoader):
        self.data = data

    def play_wav(self, file: str, noised=False):
        """
        Lit un fichier .wav et permet de l'écouter dans un notebook Jupyter.

        Args:
            file_path (str): Chemin vers le fichier .wav.

        Returns:
            Audio: Un objet audio prêt à être lu dans Jupyter.
        """
        if noised:
            freq, data = self.data.freq_x[file], self.data.data_x[file]
        else:
            freq, data = self.data.freq_y[file], self.data.data_y[file]

        print(f"Sampling frequency: {freq} Hz")
        print(f"Dimension: {data.shape}")
        return Audio(data, rate=freq)

    def display_signal(self, file: str, noised=False, figsize=(10, 4)):
        """
        Lit un fichier .wav et permet de l'écouter dans un notebook Jupyter.

        Args:
            file_path (str): Chemin vers le fichier .wav.

        Returns:
            Audio: Un objet audio prêt à être lu dans Jupyter.
        """
        if noised:
            freq, data = self.data.freq_x[file], self.data.data_x[file]
        else:
            freq, data = self.data.freq_y[file], self.data.data_y[file]

        n_samples = len(data)
        time = np.linspace(0, n_samples / freq, n_samples)
        post_fix_title = "Noisy" if noised else "Original"
        # Affichage
        plt.figure(figsize=figsize)
        plt.plot(time, data)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"{post_fix_title} Signal {file}")
        plt.grid()
        plt.show()

    def display_spectrogram(
        self,
        file: str,
        noised=False,
        nperseg: int = 2**13,
        figsize=(10, 4),
        vbounds: None | tuple[float] = None,
    ):
        """
        Affiche le spectrogramme

        Args:
            file (str): nom du fichier
            noised (bool, optional): bruité ou non. Defaults to False.
            figsize (tuple, optional): taille de la figure. Defaults to (10, 4).
        """

        if noised:
            freq, signal = self.data.freq_x[file], self.data.data_x[file]
        else:
            freq, signal = self.data.freq_y[file], self.data.data_y[file]

        # Calcul du spectrogramme
        f_stft, t_spect, Sxx = stft(
            signal, fs=freq, nperseg=nperseg, noverlap=nperseg // 2, window="hamming"
        )
        post_fix_title = "Noisy" if noised else "Original"

        # Visualisation du spectrogramme
        plt.figure(figsize=figsize)
        if vbounds is not None:
            plt.pcolormesh(
                t_spect,
                f_stft,
                20 * np.log10(np.abs(Sxx)),
                shading="gouraud",
                vmin=vbounds[0],
                vmax=vbounds[1],
            )
        else:
            plt.pcolormesh(
                t_spect, f_stft, 20 * np.log10(np.abs(Sxx)), shading="gouraud"
            )
        plt.colorbar(label="Amplitude (dB)")
        plt.title(f"Spectrogramme {post_fix_title} Signal {file}")
        plt.xlabel("Temps (s)")
        plt.ylabel("Fréquence (Hz)")
        plt.ylim(0, 100)  # Zoom sur les fréquences jusqu'à 100 Hz
        plt.tight_layout()
        plt.show()
