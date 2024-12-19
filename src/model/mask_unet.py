"""Unet model"""

# from src.libs.preprocessing import HarmonizedData

import torch
import torch.nn as nn


class UNet(nn.Module):
    """Unet"""

    def __init__(self, id_experiment: int | None):
        super(UNet, self).__init__()
        self.id_experiment = id_experiment

        # Encoder
        self.enc_conv1 = self.conv_block(1, 16, 5, stride=2)
        self.enc_conv2 = self.conv_block(16, 32, 5, stride=2)
        self.enc_conv3 = self.conv_block(32, 64, 5, stride=2)
        self.enc_conv4 = self.conv_block(64, 128, 5, stride=2)
        self.enc_conv5 = self.conv_block(128, 256, 5, stride=2)
        self.enc_conv6 = self.conv_block(256, 512, 5, stride=2)

        # Decoder
        self.dec_deconv1 = self.deconv_block(512, 256, 5, stride=2)
        self.dec_deconv2 = self.deconv_block(256 + 256, 128, 5, stride=2)
        self.dec_deconv3 = self.deconv_block(128 + 128, 64, 5, stride=2)
        self.dec_deconv4 = self.deconv_block(64 + 64, 32, 5, stride=2)
        self.dec_deconv5 = self.deconv_block(32 + 32, 16, 5, stride=2)
        self.dec_deconv6 = nn.ConvTranspose2d(
            16 + 16, 1, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.final_activation = nn.ReLU()

    def conv_block(self, in_channels, out_channels, kernel_size, stride=1):
        """conv"""
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def deconv_block(self, in_channels, out_channels, kernel_size, stride=1):
        """deconv"""
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                output_padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, x):
        """forward"""
        # Encoder
        conv1 = self.enc_conv1(x)
        conv2 = self.enc_conv2(conv1)
        conv3 = self.enc_conv3(conv2)
        conv4 = self.enc_conv4(conv3)
        conv5 = self.enc_conv5(conv4)
        conv6 = self.enc_conv6(conv5)

        # Decoder
        deconv1 = self.dec_deconv1(conv6)
        deconv2 = self.dec_deconv2(torch.cat([deconv1, conv5], dim=1))
        deconv3 = self.dec_deconv3(torch.cat([deconv2, conv4], dim=1))
        deconv4 = self.dec_deconv4(torch.cat([deconv3, conv3], dim=1))
        deconv5 = self.dec_deconv5(torch.cat([deconv4, conv2], dim=1))
        deconv6 = self.dec_deconv6(torch.cat([deconv5, conv1], dim=1))
        output = self.final_activation(deconv6)

        return output


if __name__ == "__main__":
    # Exemple d'utilisation
    data_train = {
        "x": torch.randn(100, 512, 128),  # 100 exemples d'entrée
        "y": torch.randn(100, 512, 128),  # 100 exemples de vérité terrain
    }

    # Initialisation du modèle
    model = UNet(id_experiment=0)
