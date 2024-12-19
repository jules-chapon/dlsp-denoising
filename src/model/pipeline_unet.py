"""pipeline unet"""

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch

from src.model.pipeline import Pipeline
from src.model.mask_unet import UNet


class PipelineUnet(Pipeline):
    """Pipeline UNET"""

    def __init__(self, id_experiment: int | None):
        super(Pipeline, self).__init__(id_experiment)
        self.model = UNet(id_experiment=id_experiment)

    def full_pipeline(self, data_train, data_test):
        raise NotImplementedError

    def learning_pipeline(self, data_train, data_test):
        raise NotImplementedError

    def testing_pipeline(self, data_train, data_test):
        raise NotImplementedError


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
    PipelineUnet(id_experiment=1)
