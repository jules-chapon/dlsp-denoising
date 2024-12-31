"""Pipeline for Wave-U-Net models"""

import time
import os
import pickle as pkl
import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.libs.preprocessing import (
    HarmonizedData,
    from_numpy_to_torch,
    from_torch_to_numpy,
)

from src.configs import constants, names

from src.model.pipeline import Pipeline
from src.model.waveunet import WaveUNet


class PipelineWaveUnet(Pipeline):
    """Pipeline WaveUNET"""

    def __init__(self, id_experiment: int | None = None):
        super().__init__(id_experiment)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.folder_name = f"waveunet_{id_experiment}"
        os.makedirs(
            os.path.join(constants.OUTPUT_FOLDER, self.folder_name), exist_ok=True
        )
        print(f"Device: {self.device}")
        self.model = WaveUNet(id_experiment=id_experiment)
        self.model.to(self.device)

    def full_pipeline(self, data_train: HarmonizedData, data_test: HarmonizedData):
        dataloader_train, dataloader_valid = self.get_dataloader(data_train=data_train)
        print("Dataloaders are okay")
        train_loss, valid_loss = self.train(
            dataloader_train=dataloader_train, dataloader_valid=dataloader_valid
        )
        del dataloader_train, dataloader_valid
        self.save()
        self.save_losses(train_loss=train_loss, valid_loss=valid_loss)
        print("Model saved")
        tensor_noised_test = from_numpy_to_torch(array=data_test.x)
        tensor_denoised_test = self.predict(inputs=tensor_noised_test)
        array_denoised_test = from_torch_to_numpy(tensor=tensor_denoised_test)
        self.save_data_to_numpy(
            array_noised=data_test.x,
            array_original=data_test.y,
            array_denoised=array_denoised_test,
        )
        print("End of the pipeline")
        return None

    def learning_pipeline(self, data_train: HarmonizedData, data_test: HarmonizedData):
        raise NotImplementedError

    def testing_pipeline(self, data_train: HarmonizedData, data_test: HarmonizedData):
        raise NotImplementedError

    #####################
    #### UTILS ##########
    #####################

    def get_dataloader(
        self, data_train: HarmonizedData
    ) -> tuple[DataLoader, DataLoader]:
        array_x_train, array_x_valid, array_y_train, array_y_valid = train_test_split(
            data_train.x,
            data_train.y,
            test_size=constants.TRAIN_VALID_SPLIT,
            random_state=constants.RANDOM_SEED,
        )
        x_train = torch.tensor(array_x_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(array_y_train, dtype=torch.float32).to(self.device)
        x_valid = torch.tensor(array_x_valid, dtype=torch.float32).to(self.device)
        y_valid = torch.tensor(array_y_valid, dtype=torch.float32).to(self.device)
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "Number of inputs and targets in the train set must be the same."
            )
        else:
            print(f"Number of training inputs: {x_train.shape[0]}")
        if x_valid.shape[0] != y_valid.shape[0]:
            raise ValueError(
                "Number of inputs and targets in the valid set must be the same."
            )
        else:
            print(f"Number of validation inputs: {x_valid.shape[0]}")
        train_dataset = TensorDataset(
            x_train.reshape([x_train.shape[0], 1, x_train.shape[1]]),
            y_train.reshape([y_train.shape[0], 1, y_train.shape[1]]),
        )
        valid_dataset = TensorDataset(
            x_valid.reshape([x_valid.shape[0], 1, x_valid.shape[1]]),
            y_valid.reshape([y_valid.shape[0], 1, y_valid.shape[1]]),
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.model.params[names.BATCH_SIZE], shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=self.model.params[names.BATCH_SIZE], shuffle=True
        )
        del (
            array_x_train,
            array_y_train,
            array_x_valid,
            array_y_valid,
            train_dataset,
            valid_dataset,
        )
        return train_dataloader, valid_dataloader

    #####################
    #### TRAIN PART #####
    #####################

    def train(self, dataloader_train: DataLoader, dataloader_valid: DataLoader):
        # Définition de la fonction de perte et de l'optimiseur
        criterion = nn.MSELoss()  # Perte adaptée pour une tâche de régression
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.model.params[names.LEARNING_RATE],
            betas=self.model.params[names.BETAS],
        )
        # Historique des pertes
        train_loss_history = []
        valid_loss_history = []
        # Boucle d'entraînement
        for epoch in range(self.model.params[names.NB_EPOCHS]):
            t0 = time.time()
            self.model.train()
            train_loss = 0.0
            for inputs, targets in dataloader_train:
                inputs, targets = (
                    inputs.to(self.device),
                    targets.to(self.device),
                )  # (B, 1, T) -> (B, 1, T)
                # Réinitialise les gradients
                optimizer.zero_grad()
                # Passe avant
                outputs = self.model(inputs)
                # Calcul de la perte
                loss = criterion(outputs, targets)
                # Rétropropagation
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            # Validation
            self.model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for inputs, targets in dataloader_valid:
                    inputs, targets = (
                        inputs.to(self.device),
                        targets.to(self.device),
                    )  # (B, 1, T) -> (B, 1, T)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    valid_loss += loss.item()
            # Moyenne des pertes sur l'époque
            train_loss = train_loss / len(dataloader_train)
            valid_loss = valid_loss / len(dataloader_valid)
            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)
            print(
                f"Epoch [{epoch + 1}/{self.model.params[names.NB_EPOCHS]}]. Train Loss: {train_loss:.4f}. Valid Loss: {valid_loss:.4f}. Time: {time.time()-t0} (s)"
            )
        print("Training is over.")
        return train_loss_history, valid_loss_history

    def predict(self, inputs: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)  # inputs: (B, 1, T)
            outputs = self.model(inputs)  # outputs (B, 1, T)
        return outputs

    def save(self):
        """save model"""
        path = os.path.join(constants.OUTPUT_FOLDER, self.folder_name, "pipeline.pkl")
        with open(path, "wb") as file:
            pkl.dump(self, file)

    def save_losses(self, train_loss: list[float], valid_loss: list[float]) -> None:
        """save loss"""
        np.save(
            os.path.join(constants.OUTPUT_FOLDER, self.folder_name, "train_loss.npy"),
            train_loss,
        )
        np.save(
            os.path.join(constants.OUTPUT_FOLDER, self.folder_name, "valid_loss.npy"),
            valid_loss,
        )

    def save_data_to_numpy(
        self,
        array_noised: np.ndarray,
        array_original: np.ndarray,
        array_denoised: np.ndarray,
    ) -> None:
        np.save(
            os.path.join(constants.OUTPUT_FOLDER, self.folder_name, "array_noised.npy"),
            array_noised,
        )
        np.save(
            os.path.join(
                constants.OUTPUT_FOLDER, self.folder_name, "array_original.npy"
            ),
            array_original,
        )
        np.save(
            os.path.join(
                constants.OUTPUT_FOLDER, self.folder_name, "array_denoised.npy"
            ),
            array_denoised,
        )


# if __name__ == "__main__":
#     # Import data
#     path_train_x = "data/input/denoising/train_small"
#     path_train_y = "data/input/voice_origin/train_small"
#     from src.libs import preprocessing

#     data_loader = preprocessing.DataLoader(path_x=path_train_x, path_y=path_train_y)
#     harmonized_data = data_loader.get_harmonized_data(downsample=True)
#     clean = True
#     if clean:
#         n_reduction = 1000
#         harmonized_data.x = harmonized_data.x[:n_reduction]
#         harmonized_data.y = harmonized_data.y[:n_reduction]
#         harmonized_data.names = harmonized_data.names[:n_reduction]
#         harmonized_data.n_samples = n_reduction
#         del data_loader
#     # Create pipeline
#     pipeline = PipelineWaveUnet(id_experiment=200)
#     pipeline.full_pipeline(harmonized_data, harmonized_data)
