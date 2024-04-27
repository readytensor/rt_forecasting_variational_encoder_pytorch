import os
import warnings
import math
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader


from prediction.vae_model_conv import VariationalAutoencoderConv as VAE
from logger import get_logger

logger = get_logger(__name__)

# Check for GPU availability
logger.info(f"cuda available: {torch.cuda.is_available()}")

MODEL_PARAMS_FNAME = "model_params.save"
MODEL_WTS_FNAME = "model_wts.save"
COST_THRESHOLD = float("inf")


def get_patience_factor(N):
    # magic number - just picked through trial and error
    patience = max(4, int(38 - math.log(N, 1.5)))
    return patience


class CustomDataset(Dataset):
    # Handles input-output pair loading
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        if self.y is not None:
            return self.x[index], self.y[index]
        return self.x[index]

    def __len__(self):
        return len(self.x)


class Forecaster:
    MODEL_NAME = "VAE_Timeseries_Forecaster"

    def __init__(
        self,
        encode_len,
        decode_len,
        feat_dim,
        latent_dim,
        first_hidden_dim,
        second_hidden_dim,
        third_hidden_dim,
        reconstruction_wt=100.0,
        lr=1e-3,
        **kwargs,
    ):
        self.encode_len = encode_len
        self.decode_len = decode_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = [
            int(first_hidden_dim),
            int(second_hidden_dim),
            int(third_hidden_dim),
        ]
        self.reconstruction_wt = reconstruction_wt
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VAE(
            encode_len=encode_len,
            decode_len=decode_len,
            feat_dim=feat_dim,
            latent_dim=latent_dim,
            hidden_layer_sizes=self.hidden_layer_sizes,
            reconstruction_wt=reconstruction_wt,
        ).to(self.device)

        self.criterion = MSELoss(reduction="sum")
        self.optimizer = None #set later
        self.scheduler = None #set later

    def _train_on_data(
        self,
        train_data: np.ndarray,
        valid_data: np.ndarray = None,
        max_epochs: int = 250,
        verbose: int = 1,
    ):
        train_X, train_y = self._get_X_and_y(train_data, is_train=True)
        batch_size = max(1, min(train_X.shape[0] // 8, 256))
        train_dataset = CustomDataset(train_X, train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        valid_loader = None
        if valid_data is not None:
            valid_X, valid_y = self._get_X_and_y(valid_data, is_train=True)
            valid_dataset = CustomDataset(valid_X, valid_y)
            valid_loader = DataLoader(
                dataset=valid_dataset, batch_size=batch_size, shuffle=False
            )

        best_loss = float("inf")
        patience = get_patience_factor(len(train_X))
        logger.info(f"Patience for early stopping = {patience}")

        # reset the optimizer otherwise re-training slows down
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5,
            patience=patience//2, verbose=True)

        patience_counter = 0  # Counter for early stopping
        for epoch in range(max_epochs):
            self.model.train()
            train_total_loss = 0
            train_reconstruction_loss = 0
            train_kl_loss = 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                reconstruction, z_mean, z_log_var = self.model(X)
                reconstruction_loss = self.criterion(reconstruction, y)                
                kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                loss = self.reconstruction_wt * reconstruction_loss + kl_loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.info("Cost is inf or NaN, so stopping training!!")
                    return  # Exit the training loop
                loss.backward()
                self.optimizer.step()
                train_total_loss += loss.item()
                train_reconstruction_loss += reconstruction_loss.item()
                train_kl_loss += kl_loss.item()

            # Calculate averages of losses
            num_batches = len(train_loader)
            avg_train_total_loss = train_total_loss / num_batches
            avg_train_reconstruction_loss = train_reconstruction_loss / num_batches
            avg_train_kl_loss = train_kl_loss / num_batches

            if verbose:
                logger.info(f"Epoch [{epoch+1}/{max_epochs}]: ")
                logger.info(f"Train Reconst. Loss: {avg_train_reconstruction_loss:.4f}  "
                      f"Train KL Loss: {avg_train_kl_loss:.4f}  "
                      f"Train Total Loss: {avg_train_total_loss:.4f}")

            if valid_loader:
                self.model.eval()
                val_total_loss = 0
                val_reconstruction_loss = 0
                val_kl_loss = 0
                with torch.no_grad():
                    for X, y in valid_loader:
                        X, y = X.to(self.device), y.to(self.device)
                        reconstruction, z_mean, z_log_var = self.model(X)
                        reconstruction_loss = self.criterion(reconstruction, y)
                        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss
                        val_total_loss += total_loss.item()
                        val_reconstruction_loss += reconstruction_loss.item()
                        val_kl_loss += kl_loss.item()

                # Calculate averages of validation losses
                num_batches = len(valid_loader)
                avg_val_total_loss = val_total_loss / num_batches
                avg_val_reconstruction_loss = val_reconstruction_loss / num_batches
                avg_val_kl_loss = val_kl_loss / num_batches

                # Scheduler step is called after validating
                self.scheduler.step(avg_val_total_loss)

                if verbose:
                    logger.info(
                        f"Valid Reconst. Loss: {avg_val_reconstruction_loss:.4f}  "
                        f"Valid KL Loss: {avg_val_kl_loss:.4f}  "
                        f"Valid Total Loss: {avg_val_total_loss:.4f}")

                if avg_val_total_loss < best_loss:
                    best_loss = avg_val_total_loss
                    patience_counter = 0  # reset the patience counter
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            logger.info(
                                f"Early stopping triggered after {epoch+1} epochs"
                                " due to no improvement in validation loss."
                            )
                        break
            else:
                # Scheduler step is called after validating
                self.scheduler.step(avg_train_total_loss)
                if avg_train_total_loss < best_loss:
                    best_loss = avg_train_total_loss
                    logger.info(f"best_loss={best_loss:.4f}")
                    patience_counter=0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            logger.info(
                                f"Early stopping triggered after {epoch+1} epochs"
                                " due to no improvement in training loss."
                            )
                        break

    def fit(self, train_data, valid_data, pre_training_data=None, max_epochs=1000, verbose=1):
        if pre_training_data is not None:
            logger.info("Conducting pretraining...")
            self._train_on_data(
                train_data=torch.tensor(pre_training_data.astype(np.float32), dtype=torch.float32),
                valid_data=None,
                verbose=verbose,
                max_epochs=max_epochs,
            )
        logger.info("Training on main data...")
        train_data = torch.tensor(train_data.astype(np.float32), dtype=torch.float32)
        if valid_data is not None:
            valid_data = torch.tensor(valid_data.astype(np.float32), dtype=torch.float32)
        self._train_on_data(
            train_data=train_data,
            valid_data=valid_data,
            verbose=verbose,
            max_epochs=max_epochs,
        )

    def _get_X_and_y(self, data, is_train=True):
        if is_train:
            return data[:, : self.encode_len, :], data[:, self.encode_len :, :1]
        return data[:, -self.encode_len :, :], None

    def predict(self, data):
        self.model.eval()
        X, y = self._get_X_and_y(data, is_train=False)
        X = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            reconstruction, _, _ = self.model(X)
        return reconstruction.cpu().numpy()

    def evaluate(self, test_data):
        self.model.eval()
        test_X, _ = self._get_X_and_y(test_data, is_train=False)
        test_X = torch.FloatTensor(test_X).to(self.device)
        with torch.no_grad():
            reconstruction, _, _ = self.model(test_X)
            reconstruction_loss = self.criterion(reconstruction, test_X)
        return reconstruction_loss.item()

    def save(self, model_path):
        model_params = {
            "encode_len": self.encode_len,
            "decode_len": self.decode_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "first_hidden_dim": self.hidden_layer_sizes[0],
            "second_hidden_dim": self.hidden_layer_sizes[1],
            "third_hidden_dim": self.hidden_layer_sizes[2],
            "reconstruction_wt": self.reconstruction_wt,
            "lr": self.lr,
        }
        joblib.dump(model_params, os.path.join(model_path, MODEL_PARAMS_FNAME))
        torch.save(self.model.state_dict(), os.path.join(model_path, MODEL_WTS_FNAME))

    @classmethod
    def load(cls, model_path):
        model_params = joblib.load(os.path.join(model_path, MODEL_PARAMS_FNAME))
        forecaster = cls(**model_params)
        forecaster.model.load_state_dict(
            torch.load(os.path.join(model_path, MODEL_WTS_FNAME))
        )
        return forecaster

    def __str__(self):
        return f"Model name: {self.MODEL_NAME}"


def train_predictor_model(
    train_data: np.ndarray,
    valid_data: np.ndarray,
    forecast_length: int,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the forecaster model.

    Args:
        history (np.ndarray): The training data inputs.
        forecast_length (int): Length of forecast window.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """
    model = Forecaster(
        encode_len=train_data.shape[1]-forecast_length,
        decode_len=forecast_length,
        feat_dim=train_data.shape[2],
        **hyperparameters,
    )
    model.fit(
        train_data=train_data,
        valid_data=valid_data,
        pre_training_data=None,
    )
    return model


def predict_with_model(model: Forecaster, test_data: np.ndarray) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (np.ndarray): The test input data for forecasting.

    Returns:
        np.ndarray: The forecast.
    """
    return model.predict(test_data)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
