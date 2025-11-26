import torch
import torch.nn as nn
import pytorch_lightning as pl

# --- 1. FUNCIÓN DE PÉRDIDA PERSONALIZADA (V9 - Centered) ---
class DirectionalMSELoss(nn.Module):
    def __init__(self, penalty_factor=10.0, stagnation_penalty=1.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none') 
        self.penalty_factor = penalty_factor
        self.stagnation_penalty = stagnation_penalty 

    def forward(self, y_pred, y_true):
        # 1. Error base (MSE) - Este no cambia, mide distancia pura
        loss = self.mse(y_pred, y_true)
        
        # --- CORRECCIÓN CRÍTICA: RECENTRADO ---
        # Como los datos están entre [0, 1], el "0 real" (sin cambio) no es 0.
        # Asumimos que la media del batch representa el punto neutral aproximado.
        # Esto convierte los valores [0, 1] en aprox [-0.5, +0.5]
        batch_center = y_true.mean()
        
        pred_centered = y_pred - batch_center
        true_centered = y_true - batch_center
        
        # 2. Penalización por Dirección (Ahora sí funciona el signo)
        # Si uno está por encima de la media y el otro por debajo -> Signos opuestos
        direction_match = torch.sign(pred_centered) * torch.sign(true_centered)
        dir_penalty = torch.where(direction_match < 0, self.penalty_factor, 1.0)
        
        # 3. Penalización por "Flatline" (Estancamiento)
        # Ahora castigamos si 'pred_centered' se acerca a 0 (que es la media del batch)
        # Es decir, castigamos si predice el promedio aburrido.
        flat_penalty = torch.exp(-torch.abs(pred_centered) * 50) * self.stagnation_penalty
        
        # Combinación
        total_loss = (loss * dir_penalty) + flat_penalty
        
        return torch.mean(total_loss)
    
# --- 1. BLOQUE ESPECIAL PARA CNN-LSTM ---
# Necesitamos esto porque nn.Sequential no sabe girar dimensiones
class CNNLSTM_Block(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        
        # A. La CNN 
        # Extrae 64 características visuales de la serie temporal
        self.cnn_out_channels = 64
        self.conv = nn.Conv1d(
            in_channels=input_size, 
            out_channels=self.cnn_out_channels, 
            kernel_size=3, 
            padding=1 # Padding=1 mantiene el largo de la secuencia igual
        )
        self.relu = nn.ReLU()
        
        # B. La LSTM (El Cerebro)
        # Recibe lo que sale de la CNN (64 canales)
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        # x original: (Batch, 30 días, 19 features)
        
        # 1. Girar para la CNN: (Batch, 19, 30)
        x = x.permute(0, 2, 1)
        
        # 2. Aplicar CNN
        x = self.conv(x)
        x = self.relu(x)
        
        # 3. Girar de vuelta para la LSTM: (Batch, 30, 64)
        x = x.permute(0, 2, 1)
        
        # 4. Aplicar LSTM
        # Devolvemos la salida y el estado oculto para cumplir con el estándar de LSTMRegressor
        out, hidden = self.lstm(x)
        return out, hidden

# --- 2. REGRESOR PRINCIPAL (LIGHTNING) ---
class LSTMRegressor(pl.LightningModule):
    def __init__(self, input_size=19, model_name='LSTM', hidden_size=64, num_layers=2, dropout=0.2, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        # Usamos la Fábrica para construir el corazón del modelo
        self.sequence_model, self.regressor = ModelFactory(
            model_name, 
            self.hparams
        )
        
        #self.criterion = DirectionalMSELoss(penalty_factor=10.0)
        #self.criterion = nn.MSELoss()
        self.criterion = DirectionalMSELoss(penalty_factor=10.0, stagnation_penalty=0.5)

    def forward(self, x):
        # Paso 1: Procesar secuencia (LSTM, GRU o CNN-LSTM)
        # output shape: (batch, seq_len, hidden_size)
        seq_out, _ = self.sequence_model(x) 
        
        # Paso 2: Coger solo el último día
        last_time_step = seq_out[:, -1, :]
        
        # Paso 3: Regresión final
        prediction = self.regressor(last_time_step)
        
        return prediction

    def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            self.log("train_loss", loss, prog_bar=True)
            return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # 1. El error de tu modelo
        model_loss = self.criterion(y_hat, y)
        
        # 2. El error de la "IA Tonta" (CORREGIDO)
        # En lugar de predecir ceros, predice la MEDIA del batch.
        # Esto equivale a decir: "No sé qué pasará, así que apuesto al promedio".
        # Si tu modelo no supera al promedio, no sirve.
        batch_mean = torch.mean(y)
        naive_target = torch.full_like(y, batch_mean) # Llenamos con la media (aprox 0.5)
        
        naive_loss = torch.nn.functional.mse_loss(naive_target, y)
        
        # 3. SKILL SCORE (R2 Score aproximado)
        skill_score = 1.0 - (model_loss / (naive_loss + 1e-8))
        
        self.log("val_loss", model_loss, prog_bar=True, on_epoch=True)
        self.log("val_skill", skill_score, prog_bar=True, on_epoch=True)
        
        return model_loss
        
    def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            self.log("test_loss", loss, prog_bar=True)
            return loss

    def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)

# --- 3. FÁBRICA DE MODELOS ---

def create_lstm_model(input_size, hidden_size, num_layers, dropout):
    lstm = nn.LSTM(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout if num_layers > 1 else 0, batch_first=True
    )
    return lstm, nn.Linear(hidden_size, 1)

def create_gru_model(input_size, hidden_size, num_layers, dropout):
    gru = nn.GRU(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout if num_layers > 1 else 0, batch_first=True
    )
    return gru, nn.Linear(hidden_size, 1)

def create_cnn_lstm_model(input_size, hidden_size, num_layers, dropout):
    # Instanciamos nuestro bloque personalizado
    block = CNNLSTM_Block(input_size, hidden_size, num_layers, dropout)
    # El regresor sigue recibiendo 'hidden_size' porque es lo que escupe la parte LSTM del bloque
    return block, nn.Linear(hidden_size, 1)

def ModelFactory(model_name: str, hparams: dict):
    """
    Selector inteligente de arquitectura.
    """
    p = hparams # Alias corto
    
    if model_name == 'LSTM':
        return create_lstm_model(p['input_size'], p['hidden_size'], p['num_layers'], p['dropout'])
    
    elif model_name == 'GRU':
        return create_gru_model(p['input_size'], p['hidden_size'], p['num_layers'], p['dropout'])
    
    elif model_name == 'CNNLSTM': # <--- NUEVO CASO
        return create_cnn_lstm_model(p['input_size'], p['hidden_size'], p['num_layers'], p['dropout'])
    
    else:
        raise ValueError(f"Modelo {model_name} no reconocido. Use 'LSTM', 'GRU' o 'CNNLSTM'.")