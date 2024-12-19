import torch
import torch.nn as nn

from src.libs.preprocessing import HarmonizedData

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class UNet(nn.Module):
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

    # def full_pipeline(self, data_train: HarmonizedData, data_test: HarmonizedData): ...

    # def train(self, data_train: HarmonizedData): ...


def train_unet(
    model, data_train, epochs=10, batch_size=32, learning_rate=0.001, device="cuda"
):
    """
    Entraîne le modèle UNet avec les données fournies.

    Args:
        model (torch.nn.Module): Le modèle UNet.
        data_train (dict): Contient 'x' (entrées) et 'y' (vérités terrain).
        epochs (int): Nombre d'époques d'entraînement.
        batch_size (int): Taille des lots.
        learning_rate (float): Taux d'apprentissage.
        device (str): Dispositif pour l'entraînement ('cuda' ou 'cpu').

    Returns:
        list: Historique des pertes d'entraînement.
    """
    # Déplace le modèle vers l'appareil spécifié
    model = model.to(device)

    # Préparation des données
    x_train = torch.tensor(data_train["x"], dtype=torch.float32)
    y_train = torch.tensor(data_train["y"], dtype=torch.float32)

    # Ajout d'une dimension canal pour les données
    x_train = x_train.unsqueeze(1)  # (N, 1, H, W)
    y_train = y_train.unsqueeze(1)  # (N, 1, H, W)

    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Définition de la fonction de perte et de l'optimiseur
    criterion = nn.L1Loss()  # Perte adaptée pour une tâche de régression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Historique des pertes
    loss_history = []

    # Boucle d'entraînement
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Réinitialise les gradients
            optimizer.zero_grad()

            # Passe avant
            outputs = model(inputs)

            # Calcul de la perte
            loss = criterion(outputs * inputs, targets)

            # Rétropropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Moyenne des pertes sur l'époque
        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)

        print(f"Époque [{epoch + 1}/{epochs}], Perte: {epoch_loss:.4f}")

    print("Entraînement terminé.")
    return loss_history


if __name__ == "__main__":
    # Exemple d'utilisation
    data_train = {
        "x": torch.randn(100, 512, 128),  # 100 exemples d'entrée
        "y": torch.randn(100, 512, 128),  # 100 exemples de vérité terrain
    }

    # Initialisation du modèle
    model = UNet(id_experiment=0)

    # Entraînement
    loss_history = train_unet(
        model,
        data_train,
        epochs=3,
        batch_size=16,
        learning_rate=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
