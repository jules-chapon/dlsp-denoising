"""Wave-U-Net class"""

import torch
import torch.nn as nn

from src.configs import ml_config, names
import torch.nn.functional as F


class WaveUNet(nn.Module):
    def __init__(self, id_experiment: int | None = None) -> None:
        """
        Initialize class instance.

        Args:
            id_experiment (int | None, optional): ID of the experiment. Defaults to None.
        """
        super(WaveUNet, self).__init__()
        self.id_experiment = id_experiment
        self.params = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
        ### ENCODER
        self.encoders = nn.ModuleList()
        for i in range(self.params[names.DEPTH]):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.params[names.NB_CHANNELS_INPUT]
                        if i == 0
                        else self.params[names.NB_FILTERS] * (2 ** (i - 1)),
                        self.params[names.NB_FILTERS] * (2**i),
                        kernel_size=15,
                        stride=1,
                        padding=7,
                    ),
                    nn.LeakyReLU(),
                )
            )
        ### BOTTLENECK
        self.bottleneck = nn.Sequential(
            nn.Conv1d(
                self.params[names.NB_FILTERS] * (2 ** (self.params[names.DEPTH] - 1)),
                self.params[names.NB_FILTERS] * (2 ** self.params[names.DEPTH]),
                kernel_size=15,
                stride=1,
                padding=7,
            ),
            nn.LeakyReLU(),
        )
        ### DECODER
        self.decoders = nn.ModuleList()
        for i in range(self.params[names.DEPTH] - 1, -1, -1):
            self.decoders.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.params[names.NB_FILTERS] * (2 ** (i + 1) + 2**i),
                        self.params[names.NB_FILTERS] * (2**i),
                        kernel_size=5,
                        stride=1,
                        padding=2,
                    ),
                    nn.LeakyReLU(),
                )
            )
        ### FINAL LAYER
        self.final_conv = nn.Sequential(
            nn.Conv1d(
                self.params[names.NB_FILTERS] + self.params[names.NB_CHANNELS_INPUT],
                self.params[names.NB_CHANNELS_OUTPUT],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor (B, 1, T).

        Returns:
            torch.Tensor: Output tensor : denoised signal (B, 1, T).
        """
        skips = [x]
        ### ENCODER
        for encoder in self.encoders:
            x = encoder(x)  # (B, C', T)
            skips.append(x)
            x = x[:, :, ::2]  # Downsampling : (B, C', T'=T/2)

        ### BOTTLENECK
        x = self.bottleneck(x)  # (B, C'', T'')

        ### DECODER
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]  # (B, C', T')
            x = F.interpolate(
                x, size=skip.shape[2], mode="linear", align_corners=True
            )  # Upsampling : (B, C'', T' = 2*T'')
            x = torch.cat((skip, x), dim=1)  # Skip connection : (B, C' + C'', T')
            x = decoder(x)  # (B, C', T')

        ### FINAL LAYER
        x = torch.cat((skips[0], x), dim=1)  # Skip connection : (B, C, T)
        output = self.final_conv(x)  # (B, 1, T)
        return output


if __name__ == "__main__":
    # Exemple d'utilisation
    data_train = {
        "x": torch.randn(100, 1, 1000),  # 100 exemples d'entrée
        "y": torch.randn(100, 1, 1000),  # 100 exemples de vérité terrain
    }
    input_tens = data_train["x"][0][None, :]
    # Initialisation du modèle
    model = WaveUNet(id_experiment=200)
    model.eval()
    output_first = model(input_tens)
    # Exemple : sauvegarder le modèle
    torch.save(model.state_dict(), "Waveunet_model.pth")
    print("- Saved!")
    model_loaded = WaveUNet(
        id_experiment=200
    )  # Assurez-vous que la classe WaveUNet est définie ou importée
    model_loaded.eval()
    # Charger les poids sauvegardés
    model_loaded.load_state_dict(torch.load("waveunet_model.pth", weights_only=True))
    output_loaded = model_loaded(input_tens)
    print("Loaded")
    print("Difference:", torch.abs(output_first - output_loaded).max())
