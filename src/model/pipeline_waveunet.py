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
            os.path.join(constants.OUTPUT_FOLDER, self.folder_name, "training"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(constants.OUTPUT_FOLDER, self.folder_name, "testing"),
            exist_ok=True,
        )
        print(f"Device: {self.device}")
        self.model = WaveUNet(id_experiment=id_experiment).to(self.device)

    # def full_pipeline(self, data_train: HarmonizedData, data_test: HarmonizedData):
    #     dataloader_train, dataloader_valid = self.get_train_dataloader(
    #         data_train=data_train
    #     )
    #     print("Dataloaders are okay")
    #     train_loss, valid_loss = self.train(
    #         dataloader_train=dataloader_train, dataloader_valid=dataloader_valid
    #     )
    #     del dataloader_train, dataloader_valid, data_train
    #     torch.cuda.empty_cache()
    #     self.save_array_to_numpy(array=move_to_cpu(data_test.x), name="array_noised")
    #     self.save_array_to_numpy(array=move_to_cpu(data_test.y), name="array_original")
    #     print("Arrays saved")
    #     tensor_noised_test = from_numpy_to_torch(array=data_test.x)
    #     del data_test
    #     tensor_denoised_test = self.predict(inputs=tensor_noised_test)
    #     del tensor_noised_test
    #     torch.cuda.empty_cache()
    #     print("Memory free")
    #     array_denoised_test = from_torch_to_numpy(tensor=tensor_denoised_test)
    #     self.save()
    #     self.save_losses(train_loss=train_loss, valid_loss=valid_loss)
    #     print("Model saved")
    #     self.save_array_to_numpy(
    #         array=move_to_cpu(array_denoised_test), name="array_denoised"
    #     )
    #     print("End of the pipeline")
    #     return None

    def full_pipeline(self, data_train: HarmonizedData, data_test: HarmonizedData):
        self.learning_pipeline(data_train=data_train, data_test=data_test)
        self.testing_pipeline(data_train=data_train, data_test=data_test)

    def learning_pipeline(self, data_train: HarmonizedData, data_test: HarmonizedData):
        dataloader_train, dataloader_valid = self.get_train_dataloader(
            data_train=data_train
        )
        print("Dataloaders are okay")
        train_loss, valid_loss = self.train(
            dataloader_train=dataloader_train, dataloader_valid=dataloader_valid
        )
        self.save()
        self.save_losses(train_loss=train_loss, valid_loss=valid_loss)
        print("Model saved")
        print("End of the learning pipeline")

    def testing_pipeline(self, data_train: HarmonizedData, data_test: HarmonizedData):
        test_dataloader = self.get_test_dataloader(data_test=data_test)
        print("Dataloader is okay")
        noised, original, predictions = self.predict(test_dataloader=test_dataloader)
        del test_dataloader
        self.save_array_to_numpy(array=move_to_cpu(noised), name="array_noised")
        self.save_array_to_numpy(array=move_to_cpu(original), name="array_original")
        self.save_array_to_numpy(array=move_to_cpu(predictions), name="array_denoised")
        print("Arrays saved")
        print("End of the testing pipeline")

    #####################
    #### UTILS ##########
    #####################

    def get_train_dataloader(
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

    def get_test_dataloader(self, data_test: HarmonizedData) -> DataLoader:
        x_test = torch.tensor(data_test.x, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(data_test.y, dtype=torch.float32).to(self.device)
        if x_test.shape[0] != y_test.shape[0]:
            raise ValueError(
                "Number of inputs and targets in the test set must be the same."
            )
        else:
            print(f"Number of testing inputs: {x_test.shape[0]}")
        test_dataset = TensorDataset(
            x_test.reshape([x_test.shape[0], 1, x_test.shape[1]]),
            y_test.reshape([y_test.shape[0], 1, y_test.shape[1]]),
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.model.params[names.BATCH_SIZE], shuffle=True
        )
        del test_dataset
        return test_dataloader

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

    def predict(self, test_dataloader: DataLoader):
        noised = []
        original = []
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = (
                    inputs.to(self.device),
                    targets.to(self.device),
                )  # (B, 1, T) -> (B, 1, T)
                outputs = self.model(inputs)
                noised.append(inputs.cpu().numpy().squeeze())
                original.append(targets.cpu().numpy().squeeze())
                predictions.append(outputs.cpu().numpy().squeeze())
        noised = np.concatenate(noised)
        original = np.concatenate(original)
        predictions = np.concatenate(predictions)
        return noised, original, predictions

    def save(self):
        """save model"""
        path = os.path.join(
            constants.OUTPUT_FOLDER, self.folder_name, "training", "pipeline.pkl"
        )
        self.model.to("cpu")
        self_to_cpu = move_to_cpu(self)
        with open(path, "wb") as file:
            pkl.dump(self_to_cpu, file)

    def save_losses(self, train_loss: list[float], valid_loss: list[float]) -> None:
        """save loss"""
        train_loss = move_to_cpu(train_loss)
        valid_loss = move_to_cpu(valid_loss)
        np.save(
            os.path.join(
                constants.OUTPUT_FOLDER, self.folder_name, "training", "train_loss.npy"
            ),
            train_loss,
        )
        np.save(
            os.path.join(
                constants.OUTPUT_FOLDER, self.folder_name, "training", "valid_loss.npy"
            ),
            valid_loss,
        )

    def save_array_to_numpy(self, array: np.ndarray, name: str) -> None:
        array = move_to_cpu(array)
        np.save(
            os.path.join(
                constants.OUTPUT_FOLDER, self.folder_name, "testing", f"{name}.npy"
            ),
            array,
        )

    @classmethod
    def load(cls, id_experiment: int | None = None) -> "Pipeline":
        path = os.path.join(
            constants.OUTPUT_FOLDER,
            f"waveunet_{id_experiment}",
            "training",
            "pipeline.pkl",
        )
        with open(path, "rb") as file:
            pipeline = pkl.load(file)
        pipeline.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.model = pipeline.model.to(pipeline.device)
        return pipeline


def move_to_cpu(obj):
    if isinstance(obj, torch.nn.Module):
        return obj.to("cpu")
    elif isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {key: move_to_cpu(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(val) for val in obj]
    else:
        return obj


if __name__ == "__main__":
    # Import data
    path_train_x = "data/input/denoising/train_small"
    path_train_y = "data/input/voice_origin/train_small"
    from src.libs import preprocessing

    data_loader = preprocessing.DataLoader(path_x=path_train_x, path_y=path_train_y)
    harmonized_data = data_loader.get_harmonized_data(downsample=True)
    clean = True
    if clean:
        n_reduction = 1000
        harmonized_data.x = harmonized_data.x[:n_reduction]
        harmonized_data.y = harmonized_data.y[:n_reduction]
        harmonized_data.names = harmonized_data.names[:n_reduction]
        harmonized_data.n_samples = n_reduction
        del data_loader
    # Create pipeline
    pipeline = PipelineWaveUnet(id_experiment=200)
    pipeline.learning_pipeline(harmonized_data, harmonized_data)
    pipeline.testing_pipeline(harmonized_data, harmonized_data)
