import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as AT
import librosa
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, annotations, root_dir, has_label=True, max_length=320000, n_fft=1024, n_mels=128):
        """
        Args:
        """
        self.annotations = annotations
        self.root_dir = root_dir
        self.has_label = has_label
        self.max_length = max_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000,
            n_fft=self.n_fft,
            n_mels=self.n_mels
        )
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
        )
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 1])
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if self.has_label:
            label = torch.tensor(int(self.annotations.iloc[idx, 2]))
        
        # 길이를 max_length로 맞추기
        if waveform.size(1) > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            pad_size = self.max_length - waveform.size(1)
            waveform = F.pad(waveform, (0, pad_size))
            
        # Mel Spectrogram 생성
        log_mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = torchaudio.transforms.AmplitudeToDB()(log_mel_spec)  # dB로 변환
        
        spectrogram = self.spectrogram(waveform)
        
        log_mel_spec = torch.cat((spectrogram, log_mel_spec), dim=1)

        if self.has_label:
            return waveform, sample_rate, log_mel_spec, label
        else:
            return waveform, sample_rate, log_mel_spec
