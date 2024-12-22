"""pipeline unet"""

import time
import os
from datetime import datetime
import tqdm
import numpy as np

# import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import src.configs.ml_config as ml_config
import src.configs.constants as constants


from src.model.pipeline import Pipeline
from src.model.mask_unet import UNet


class PipelineUnet(Pipeline):
    """Pipeline UNET"""

    def __init__(self, id_experiment: int | None):
        super().__init__(id_experiment)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.post_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Device: {self.device}")
        self.model = UNet(id_experiment=id_experiment)
        self.model.to(self.device)

    def full_pipeline(self, data_train, data_test):
        # train
        train_args = {
            "dataloader_train": self.get_spect(data_train),
            "epochs": ml_config.EXPERIMENTS_CONFIGS[self.id_experiment]["epochs"],
        }
        loss = self.train(**train_args)
        self.save_model()
        self.save_loss(loss)
        return

    def learning_pipeline(self, data_train, data_test):
        raise NotImplementedError

    def testing_pipeline(self, data_train, data_test):
        raise NotImplementedError

    #####################
    #### UTILS ##########
    #####################

    def get_spect(self, data_train):
        """get the spectrogram"""
        x_train = torch.tensor(data_train.x, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(data_train.y, dtype=torch.float32).to(self.device)
        spect_x = generate_spectrograms_resized(x_train, self.device)
        spect_y = generate_spectrograms_resized(y_train, self.device)
        print("Spectogram shapes", spect_x.shape, spect_y.shape)
        dataloader_train = self.get_data_loader(spect_x, spect_y)
        return dataloader_train

    #####################
    #### TRAIN PART #####
    #####################

    def train(self, dataloader_train, lr=0.02, epochs=10):
        """train"""
        # Définition de la fonction de perte et de l'optimiseur
        criterion = nn.L1Loss()  # Perte adaptée pour une tâche de régression
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Historique des pertes
        loss_history = []
        # Boucle d'entraînement
        for epoch in range(epochs):
            t0 = time.time()
            self.model.train()
            running_loss = 0.0
            for inputs, targets in dataloader_train:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Réinitialise les gradients
                optimizer.zero_grad()
                # Passe avant
                outputs = self.model(inputs)
                # Calcul de la perte
                loss = criterion(outputs * inputs, targets)
                # Rétropropagation
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            # Moyenne des pertes sur l'époque
            epoch_loss = running_loss / len(dataloader_train)
            loss_history.append(epoch_loss)
            print(
                f"Époque [{epoch + 1}/{epochs}], Perte: {epoch_loss:.4f}. Time: {time.time()-t0} (s)"
            )
        print("Entraînement terminé.")
        return loss_history

    def get_data_loader(self, spect_x, spect_y, batchsize=32):
        """data loader"""
        # Ajout d'une dimension canal pour les données
        x_train = spect_x.unsqueeze(1)  # (N, 1, H, W)
        y_train = spect_y.unsqueeze(1)  # (N, 1, H, W)
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
        return dataloader

    def save_model(self):
        """save model"""
        model_name = f"model_unet_{self.post_name}.pth"
        path = os.path.join(constants.OUTPUT_FOLDER, model_name)
        torch.save(self.model.state_dict(), path)
        return

    def save_loss(self, loss: list[float]):
        """save loss"""
        np.save(
            os.path.join(constants.OUTPUT_FOLDER, f"loss_{self.post_name}.npy"), loss
        )
        return


def generate_spectrograms_resized(
    data, device, n_fft=1024, hop_length=312, win_length=1024
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
    # Paramètres pour la STFT
    window = torch.ones(n_fft)

    for signal in tqdm.tqdm(data):
        # Calcul du spectrogramme avec STFT
        window = torch.ones(n_fft, device=device).to(device)
        # Calcul du spectrogramme
        spec = torch.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        spec_magnitude = torch.log(
            torch.abs(spec)
        )  # Utilise la magnitude pour le spectrogramme
        # print("Magn shape", spec_magnitude.shape)
        # Convertir en image et redimensionner (512, 128)
        # spec_resized = torch.nn.functional.interpolate(
        #     spec_magnitude.unsqueeze(0).unsqueeze(
        #         0
        #     ),  # Ajouter deux dimensions pour batch et channel
        #     size=output_size,
        #     mode="bilinear",
        #     align_corners=False,
        # ).squeeze()  # Enlever les dimensions inutiles
        spec_resized = spec_magnitude[:-1, :-1]
        spectrograms.append(spec_resized)
    return torch.stack(spectrograms)
