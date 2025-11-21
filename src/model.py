import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTMRegressor(pl.LightningModule):
    def __init__(self, input_size=4, model_name='LSTM', hidden_size=64, num_layers=2, dropout=0.2, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        # 1. Creamos la arquitectura usando la Fábrica
        self.sequence_model, self.regressor = ModelFactory(
            model_name, 
            self.hparams # Pasamos todos los parámetros guardados
        )
        
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # Ahora usamos el nombre genérico self.sequence_model
        lstm_out, _ = self.sequence_model(x) 
        # x shape: (batch_size, sequence_length=30, input_size=4)
        # Extraemos la conclusión: solo la salida del ÚLTIMO paso de tiempo
        last_time_step = lstm_out[:, -1, :]
        
        prediction = self.regressor(last_time_step)
        
        return prediction

    def training_step(self, batch, batch_idx):
            x, y = batch
            self.train() # Asegurar modo entrenamiento
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            self.log("train_loss", loss, prog_bar=True)
            return loss

    def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            self.log("val_loss", loss, prog_bar=True)
            return loss
        
    def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            self.log("test_loss", loss, prog_bar=True)
            return loss

    def configure_optimizers(self):
            # Usamos AdamW con weight_decay bajo como buena práctica.
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)

def create_lstm_model(input_size: int, hidden_size: int, num_layers: int, dropout: float):
    lstm = nn.LSTM(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout if num_layers > 1 else 0, batch_first=True
    )
    return lstm, nn.Linear(hidden_size, 1)

def create_gru_model(input_size: int, hidden_size: int, num_layers: int, dropout: float):
    gru = nn.GRU(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout if num_layers > 1 else 0, batch_first=True
    )
    return gru, nn.Linear(hidden_size, 1)

def ModelFactory(model_name: str, hparams: dict):
    """
    Función maestra para seleccionar el modelo basado en el nombre.
    """
    if model_name == 'LSTM':
        # Llamamos a la fábrica de LSTM
        return create_lstm_model(hparams['input_size'], hparams['hidden_size'], hparams['num_layers'], hparams['dropout'])
    elif model_name == 'GRU':
        # Llamamos a la fábrica de GRU
        return create_gru_model(hparams['input_size'], hparams['hidden_size'], hparams['num_layers'], hparams['dropout'])
    else:
        raise ValueError(f"Modelo {model_name} no reconocido. Use 'LSTM' o 'GRU'.")