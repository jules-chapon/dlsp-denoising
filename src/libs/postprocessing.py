"""Functions for postprocessing"""

import torch
import torch.nn.functional as F

from src.model.mask_unet import UNet

import matplotlib.pyplot as plt


def get_unet(post_name: str):
    """get unet from post name experience"""

    model_loaded = UNet(
        id_experiment=0
    )  # Assurez-vous que la classe UNet est définie ou importée
    model_loaded.eval()
    # Charger les poids sauvegardés
    model_loaded.load_state_dict(
        torch.load(
            f"output/model_unet_{post_name}.pth",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )
    model_loaded.eval()

    return model_loaded


def upsample_spectrogram(spectrogram: torch.Tensor) -> torch.Tensor:
    """
    Suréchantillonner un spectrogramme de taille (512, 256) à (513, 257).
    """
    # Ajouter une ligne de zéros pour obtenir la dimension (513, 256)
    spectrogram_upsampled = F.pad(
        spectrogram, (0, 1, 0, 0), value=0
    )  # Pad sur la colonne pour obtenir (512, 257)

    # Ajouter une colonne de zéros pour obtenir la dimension (513, 257)
    spectrogram_upsampled = F.pad(
        spectrogram_upsampled, (0, 0, 0, 1), value=0
    )  # Pad sur la ligne pour obtenir (513, 257)

    return spectrogram_upsampled


def invert_spectrogram(
    spectrogram, n_fft=1024, hop_length=512, win_length=1024, window=None, length=None
):
    """invert a complex spectrogam of size (513,257)"""
    if window is None:
        window = torch.ones(n_fft)

    # Inverse STFT (reconstruction du signal à partir du spectrogramtrogramme)
    # pylint: disable=E1102
    reconstructed_signal = torch.istft(
        spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,  # Assurez-vous que la longueur est égale à celle du signal d'origine
    )
    # pylint: enable=E1102

    return reconstructed_signal


class Spectrogram:
    """spectrogram container"""

    def __init__(self, signal: torch.Tensor):
        self.signal = signal
        # Paramètres pour la STFT
        self.n_fft = 1024
        self.hop_length = 312  # Ajustez pour obtenir un spectrogramme de taille 512x512
        self.win_length = 1024
        self.window = torch.ones(self.n_fft)
        self.spectrogam = self.get_spectrogram()
        self.magnitude = self.get_magnitude()
        self.angle = self.get_angle()

    def get_spectrogram(self):
        """compute complex spectrogram"""
        return torch.stft(
            self.signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

    def get_angle(self):
        """compute ngle"""
        return torch.angle(self.spectrogam)

    def get_magnitude(self):
        """compute magnitude"""
        return torch.log(torch.abs(self.spectrogam))

    def plot_spectrogram(self):
        """plot spectrogram"""
        plt.figure(figsize=(10, 6))
        plt.imshow(
            self.magnitude.numpy(), aspect="auto", cmap="inferno", origin="lower"
        )
        plt.title("Spectrogramme (Clean) - Magnitude (log)")
        plt.colorbar(format="%+2.0f dB")
        plt.xlabel("Temps")
        plt.ylabel("Fréquence")
        plt.show()
