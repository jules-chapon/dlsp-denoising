"""Visualization functions"""

from src.libs.preprocessing import HarmonizedData

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.signal import stft
import torch
from src.libs.postprocessing import Spectrogram


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



def plot_spectrograms(test_x, denoised_output_upsample, test_y):
    """
    Affiche trois spectrogrammes côte à côte : signal bruité, signal débruité et signal cible,
    avec une échelle commune.

    Args:
        test_x (tensor): Signal bruité, de dimensions [1, 1, ...].
        denoised_output_upsample (tensor): Signal débruité, de dimensions [1, 1, ...].
        test_y (tensor): Signal cible, de dimensions [1, 1, ...].

    Returns:
        None: Affiche les spectrogrammes dans une figure matplotlib.
    """
    spectrogram_input = Spectrogram(torch.tensor(test_x))
    spectrogram_output = denoised_output_upsample.detach()[0, 0].numpy()
    spectrogram_target = Spectrogram(torch.tensor(test_y))

    all_magnitudes = [
        spectrogram_input.magnitude,
        spectrogram_output,
        spectrogram_target.magnitude,
    ]
    vmin = min(mag.min() for mag in all_magnitudes)
    vmax = max(mag.max() for mag in all_magnitudes)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Signal bruité
    im0 = axes[0].imshow(
        spectrogram_input.magnitude,
        aspect="auto",
        cmap="inferno",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Spectrogram (Input) - Magnitude (log)")
    axes[0].set_xlabel("Temps")
    axes[0].set_ylabel("Fréquence")
    fig.colorbar(im0, ax=axes[0], format="%+2.0f dB")

    # Spectrogramme de sortie
    im1 = axes[1].imshow(
        spectrogram_output,
        aspect="auto",
        cmap="inferno",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("Spectrogram (Output) - Magnitude (log)")
    axes[1].set_xlabel("Temps")
    axes[1].set_ylabel("Fréquence")
    fig.colorbar(im1, ax=axes[1], format="%+2.0f dB")

    # Spectrogramme target
    im2 = axes[2].imshow(
        spectrogram_target.magnitude,
        aspect="auto",
        cmap="inferno",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    axes[2].set_title("Spectrogram (Target) - Magnitude (log)")
    axes[2].set_xlabel("Temps")
    axes[2].set_ylabel("Fréquence")
    fig.colorbar(im2, ax=axes[2], format="%+2.0f dB")

    plt.tight_layout()
    plt.show()
