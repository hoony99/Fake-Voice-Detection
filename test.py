import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import webrtcvad
import warnings
from asteroid.models import ConvTasNet
import subprocess
warnings.filterwarnings('ignore')
import torchaudio
import os

class Config:
    SR = 48000
    N_MFCC = 13
    N_CLASSES = 2
    BATCH_SIZE = 256
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    MAX_LEN = 500
    CNN_CHANNELS = [32, 64, 128, 256]
    TRANSFORMER_DIM = 512
    TRANSFORMER_LAYERS = 6
    TRANSFORMER_HEADS = 8

CONFIG = Config()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conv_tasnet_model = ConvTasNet.from_pretrained("mpariente/ConvTasNet_WHAM_sepclean").to(device)
conv_tasnet_model.eval()

class CNNTransformerModel(nn.Module):
    def __init__(self, input_channels=1, input_height=128, input_width=500):
        super(CNNTransformerModel, self).__init__()

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

        cnn_output_height = input_height // (2 ** len(CONFIG.CNN_CHANNELS))
        cnn_output_width = input_width // (2 ** len(CONFIG.CNN_CHANNELS))
        self.cnn_output_size = cnn_output_height * cnn_output_width * CONFIG.CNN_CHANNELS[-1]

        self.linear = nn.Linear(self.cnn_output_size, CONFIG.TRANSFORMER_DIM)

        self.pos_encoder = PositionalEncoding(CONFIG.TRANSFORMER_DIM, dropout=0.1)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=CONFIG.TRANSFORMER_DIM,
            nhead=CONFIG.TRANSFORMER_HEADS,
            dim_feedforward=CONFIG.TRANSFORMER_DIM * 4,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, CONFIG.TRANSFORMER_LAYERS)

        self.fc = nn.Linear(CONFIG.TRANSFORMER_DIM, CONFIG.N_CLASSES)

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)

        x = x.view(x.size(0), -1, self.cnn_output_size)
        x = self.linear(x)

        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)

        x = torch.mean(x, dim=1)

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


def remove_noise_deepfilternet(input_path):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"{base_name}_DeepFilterNet3.wav"
    
    command = f"deepFilter {input_path} --output-dir ."
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Command output:", result.stdout.decode())
        print("Command error (if any):", result.stderr.decode())
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running deepFilter: {e}")
    
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output file {output_path} not found after noise removal.")
    
    try:
        y, sr = librosa.load(output_path, sr=CONFIG.SR)
    except Exception as e:
        raise RuntimeError(f"Error loading audio file {output_path}: {e}")
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)
    
    return y, sr

def apply_webrtc_vad(audio, sample_rate, frame_duration_ms=30, aggressive=3):
    vad = webrtcvad.Vad(aggressive)
    frame_length = int(sample_rate * (frame_duration_ms / 1000.0))
    padding = np.zeros(frame_length // 2)
    audio_padded = np.concatenate([padding, audio, padding])
    
    voice_flags = []
    for i in range(0, len(audio_padded) - frame_length, frame_length):
        frame = audio_padded[i:i+frame_length]
        frame_bytes = (frame * 32767).astype('int16').tobytes()
        voice_flags.append(vad.is_speech(frame_bytes, sample_rate))
    
    voice_flags = np.array(voice_flags)
    voice_flags = np.repeat(voice_flags, frame_length)
    return voice_flags[:len(audio)]

def post_process_vad(audio, voice_flags, sample_rate, min_speech_duration_ms=300, min_silence_duration_ms=100):
    frame_duration_ms = len(audio) * 1000 / len(voice_flags) / sample_rate
    min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
    min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)

    speech_regions = []
    start = None
    for i, flag in enumerate(voice_flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_speech_frames:
                speech_regions.append((start, i))
            start = None
    if start is not None and len(voice_flags) - start >= min_speech_frames:
        speech_regions.append((start, len(voice_flags)))

    processed_flags = np.zeros_like(voice_flags)
    for start, end in speech_regions:
        processed_flags[start:end] = 1

    return audio * processed_flags
   
def process_audio(audio_path):
    y, sr = remove_noise_deepfilternet(audio_path)
    voice_flags = apply_webrtc_vad(y, CONFIG.SR)
    y = post_process_vad(y, voice_flags, CONFIG.SR)
    if len(y) < CONFIG.N_FFT:
        y = np.pad(y, (0, CONFIG.N_FFT - len(y)))
    return y
    
def get_melspectrogram_feature(df):
    features = []
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG.SR,
        n_fft=1024,
        hop_length=256,
        n_mels=CONFIG.N_MELS
    ).to(torch.float32)
    
    for _, row in tqdm(df.iterrows()):
        y = process_audio(row['path'])        
        
        y_tensor = torch.from_numpy(y).to(torch.float32) 
        mel_spec = mel_spectrogram(y_tensor)
        mel_spec = librosa.power_to_db(mel_spec.numpy(), ref=np.max)
        
        if mel_spec.shape[1] < CONFIG.MAX_LEN:
            pad_width = CONFIG.MAX_LEN - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec = mel_spec[:, :CONFIG.MAX_LEN]
        
        features.append(mel_spec)

    return features


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

def inference(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader)):
            features = features.float().to(device)
            probs = model(features)
            predictions.extend(probs.cpu().numpy().tolist())
    return predictions


if __name__ == "__main__":
    test = pd.read_csv('./test1.csv')
    test_features = get_melspectrogram_feature(test)
    test_dataset = CustomDataset(test_features, None)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

    model = CNNTransformerModel(input_channels=1, input_height=CONFIG.N_MELS, input_width=CONFIG.MAX_LEN)
    model.load_state_dict(torch.load('20240715_model2.pth', map_location=device))
    model.to(device)

    preds = inference(model, test_loader, device)
    print("Length of preds:", len(preds))
    print("Content of preds:", preds)
    submit = pd.read_csv('./sample1.csv')
    submit.iloc[:, 1:] = preds
    submit.head()
    submit.to_csv('./res1.csv', index=False)