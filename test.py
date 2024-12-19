import torch


def generate_spectrograms_resized(
    data, n_fft=1024, hop_length=128, win_length=1024, output_size=(512, 128)
):
    """
    Génère des spectrogrammes redimensionnés à la taille spécifiée.
    Args:
        data (torch.Tensor): Tensor de forme (N, L) (ex. N signaux de longueur L=80000).
        n_fft (int): Taille de la FFT.
        hop_length (int): Décalage entre fenêtres consécutives.
        win_length (int): Longueur de la fenêtre d'analyse.
        output_size (tuple): Dimensions de sortie (freq, time), ex. (512, 128).
    Returns:
        torch.Tensor: Tensor de spectrogrammes de taille (N, 512, 128).
    """
    spectrograms = []
    for signal in data:
        # Calcul du spectrogramme avec STFT
        spec = torch.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            return_complex=True,
        )
        spec_magnitude = torch.abs(spec)  # Utilise la magnitude pour le spectrogramme

        # Convertir en image et redimensionner (512, 128)
        spec_resized = torch.nn.functional.interpolate(
            spec_magnitude.unsqueeze(0).unsqueeze(
                0
            ),  # Ajouter deux dimensions pour batch et channel
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze()  # Enlever les dimensions inutiles
        spectrograms.append(spec_resized)
    return torch.stack(spectrograms)


# Exemple d'utilisation
N = 10
data_train_x = torch.randn(N, 80000)  # Signaux simulés
data_train_y = torch.randn(N, 80000)

# Spectrogrammes générés
spectrograms_x = generate_spectrograms_resized(data_train_x)
spectrograms_y = generate_spectrograms_resized(data_train_y)

# Vérification des tailles
assert spectrograms_x.shape == (
    N,
    512,
    128,
), "La taille des spectrogrammes X n'est pas correcte"
assert spectrograms_y.shape == (
    N,
    512,
    128,
), "La taille des spectrogrammes Y n'est pas correcte"

print("Les spectrogrammes sont correctement générés !")
