"""Visualization functions"""

from src.libs.preprocessing import HarmonizedData

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.signal import stft


class DataVisualizer:
    """
    Permet de visualiser la forme d'onde, d'écouter un signal et d'afficher son spectrogramme.
    """

    def __init__(self, data: HarmonizedData):
        self.data = data

    def play_wav(self, file_number: int, noised=False):
        """
        Lit un fichier .wav et permet de l'écouter dans un notebook Jupyter.

        Args:
            file_path (str): Chemin vers le fichier .wav.

        Returns:
            Audio: Un objet audio prêt à être lu dans Jupyter.
        """
        data = self.data.x[file_number] if noised else self.data.y[file_number]

        return Audio(data, rate=self.data.sampling_freq)

    def display_signal(self, file_number: int, noised=False, figsize=(10, 4)):
        """
        Lit un fichier .wav et permet de l'écouter dans un notebook Jupyter.

        Args:
            file_path (str): Chemin vers le fichier .wav.

        Returns:
            Audio: Un objet audio prêt à être lu dans Jupyter.
        """

        data = self.data.x[file_number] if noised else self.data.y[file_number]

        n_samples = len(data)
        time = np.linspace(0, n_samples / self.data.sampling_freq, n_samples)
        post_fix_title = "Noisy" if noised else "Original"
        # Affichage
        plt.figure(figsize=figsize)
        plt.plot(time, data)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"{post_fix_title} Signal {self.data.names[file_number]}")
        plt.grid()
        plt.ylim(-1, 1)
        plt.show()

    def display_spectrogram(
        self,
        file_number: int,
        noised=False,
        nperseg: int = 2**8,
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

        data = self.data.x[file_number] if noised else self.data.y[file_number]

        # Calcul du spectrogramme
        f_stft, t_spect, Sxx = stft(
            data,
            fs=self.data.sampling_freq,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            window="hamming",
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
        plt.title(
            f"Spectrogramme {post_fix_title} Signal {self.data.names[file_number]}"
        )
        plt.xlabel("Temps (s)")
        plt.ylabel("Fréquence (Hz)")
        plt.ylim(0, 100)  # Zoom sur les fréquences jusqu'à 100 Hz
        plt.tight_layout()
        plt.show()
