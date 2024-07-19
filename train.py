import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import warnings
import random
import os

import torchaudio
import torchmetrics
from pydub import AudioSegment
import torchaudio
from torchaudio import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

warnings.filterwarnings('ignore')
device = torch.device("cuda")
torch.cuda.set_device(1)
## add scheduler & early stop & add CNN&Transformer Layers & FocalLoss
class Config:
    SR = 32000
    N_MFCC = 13
    N_CLASSES = 2
    BATCH_SIZE = 256
    N_EPOCHS = 50
    LR = 1e-5
    SEED = 42
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    MAX_LEN = 500
    CNN_CHANNELS = [32, 64, 128, 256]
    TRANSFORMER_DIM = 512
    TRANSFORMER_LAYERS = 6
    TRANSFORMER_HEADS = 8
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_DELTA = 0.001

CONFIG = Config()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG.SEED)

def add_noise(audio, noise_level):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_level * noise
    augmented_audio = np.clip(augmented_audio, -1, 1)
    return augmented_audio

def time_shift(audio, shift_max=0.2):
    shift = np.random.randint(-shift_max * len(audio), shift_max * len(audio))
    return np.roll(audio, shift)

def pitch_shift(audio, sr, n_steps):
    return librosa.effects.pitch_shift(audio, sr, n_steps=n_steps)
  
    
def get_melspectrogram_feature(df, apply_augmentation):
    features = []
    labels = []
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG.SR,
        n_fft=CONFIG.N_FFT,
        hop_length=CONFIG.HOP_LENGTH,
        n_mels=CONFIG.N_MELS
    ).to(torch.float32)
    
    for _, row in tqdm(df.iterrows()):
        wav_path = row['path']
        
        y, sr = librosa.load(wav_path, sr=CONFIG.SR)
        
        if apply_augmentation:
            if np.random.random() < 0.5:  
                y = add_noise(y, noise_level=np.random.uniform(0.01, 0.05)) # strong noise than train1
            if np.random.random() < 0.5:
                y = time_shift(y)
            if np.random.random() < 0.5: 
                y = pitch_shift(y, sr, n_steps=np.random.randint(-5, 5))
                
        y_tensor = torch.from_numpy(y).to(torch.float32) 
        mel_spec = mel_spectrogram(y_tensor)
        mel_spec = librosa.power_to_db(mel_spec.numpy(), ref=np.max)
        
        if mel_spec.shape[1] < CONFIG.MAX_LEN:
            pad_width = CONFIG.MAX_LEN - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec = mel_spec[:, :CONFIG.MAX_LEN]
        
        features.append(mel_spec)

        label = row['label']
        label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
        label_vector[0 if label == 'fake' else 1] = 1
        labels.append(label_vector)

    return features, labels
    
    
class CustomDataset(Dataset):
    def __init__(self, mel_specs, label=None):
        self.mel_specs = mel_specs
        self.label = label

    def __len__(self):
        return len(self.mel_specs)

    def __getitem__(self, index):
        mel_spec = torch.from_numpy(self.mel_specs[index]).float().unsqueeze(0)  # Add channel dimension
        if self.label is not None:
            return mel_spec, torch.tensor(self.label[index]).float()
        return mel_spec

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma 
        self.reduction = reduction 

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # probabilities
        # Focal Loss
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
            
class CNNTransformerModel(nn.Module):
    def __init__(self, input_channels=1, input_height=128, input_width=500):
        super(CNNTransformerModel, self).__init__()
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        for out_channels in CONFIG.CNN_CHANNELS:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels), 
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            in_channels = out_channels
        
        # Calculate the size after CNN layers
        cnn_output_height = input_height // (2 ** len(CONFIG.CNN_CHANNELS))
        cnn_output_width = input_width // (2 ** len(CONFIG.CNN_CHANNELS))
        self.cnn_output_size = cnn_output_height * cnn_output_width * CONFIG.CNN_CHANNELS[-1]
        
        # Linear layer to adjust dimensions for Transformer
        self.linear = nn.Linear(self.cnn_output_size, CONFIG.TRANSFORMER_DIM)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(CONFIG.TRANSFORMER_DIM, dropout=0.1)
        
        # Transformer layers with Multi-Head Attention
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=CONFIG.TRANSFORMER_DIM, 
            nhead=CONFIG.TRANSFORMER_HEADS,
            dim_feedforward=CONFIG.TRANSFORMER_DIM * 4,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, CONFIG.TRANSFORMER_LAYERS)
        
        # Output layer
        self.fc = nn.Linear(CONFIG.TRANSFORMER_DIM, CONFIG.N_CLASSES)
        
    def forward(self, x):
        # CNN layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Reshape for Transformer
        x = x.view(x.size(0), -1, self.cnn_output_size)
        x = self.linear(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer layers
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output layer
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience 
        self.delta = delta 
        self.counter = 0 
        self.best_score = None 
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

def train(model, optimizer, train_loader, val_loader, device, max_grad_norm=1.0):
    model.to(device)
    criterion = FocalLoss().to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=CONFIG.EARLY_STOPPING_PATIENCE, delta=CONFIG.EARLY_STOPPING_DELTA)
    best_val_score = 0
    best_model = None
    best_epoch = 0

    for epoch in range(1, CONFIG.N_EPOCHS+1):
        model.train()
        train_loss = []
        for features, labels in tqdm(train_loader):
            features, labels = features.float().to(device), labels.float().to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            train_loss.append(loss.item())
        
        _train_loss = np.mean(train_loss)
        
        if val_loader is not None:
            _val_loss, _val_score, _val_accuracy, _val_f1 = validation(model, criterion, val_loader, device)
            print(f'Epoch [{epoch}], Train Loss: [{_train_loss:.5f}], Val Loss: [{_val_loss:.5f}], '
                  f'Val AUC: [{_val_score:.5f}], Val Accuracy: [{_val_accuracy:.5f}], Val F1: [{_val_f1:.5f}]')
            
            scheduler.step(_val_loss)
            
            if best_val_score < _val_score:
                best_val_score = _val_score
                best_model = model.state_dict().copy()
                best_epoch = epoch
            
            early_stopping(_val_score)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            print(f'Epoch [{epoch}], Train Loss: [{_train_loss:.5f}]')
            best_model = model.state_dict().copy()
    
    print(f"Best validation score: {best_val_score:.5f} at epoch {best_epoch}")
    
    if best_model is not None:
        model.load_state_dict(best_model)
    return model

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score
    
def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader):
            features, labels = features.float().to(device), labels.float().to(device)
            probs = model(features)
            loss = criterion(probs, labels)
            val_loss.append(loss.item())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        _val_loss = np.mean(val_loss)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        auc_score = roc_auc_score(all_labels, all_probs, average='macro')
        accuracy = np.mean((all_probs > 0.5) == all_labels)
        f1 = f1_score(all_labels, (all_probs > 0.5), average='macro')
    
    return _val_loss, auc_score, accuracy, f1

if __name__ == "__main__":
    df = pd.read_csv('./train.csv')
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=CONFIG.SEED)

    train_features, train_labels = get_melspectrogram_feature(train_df, apply_augmentation=True)
    val_features, val_labels = get_melspectrogram_feature(val_df, apply_augmentation=False)

    train_dataset = CustomDataset(train_features, train_labels)
    val_dataset = CustomDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

    model = CNNTransformerModel(input_channels=1, input_height=CONFIG.N_MELS, input_width=CONFIG.MAX_LEN)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=CONFIG.LR)

    trained_model = train(model, optimizer, train_loader, val_loader, device)
    torch.save(trained_model.state_dict(), 'final_model.pth')