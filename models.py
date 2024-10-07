from scipy.io import wavfile
from scipy.signal import stft
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.signal import stft, istft


N_CHANNELS = 1

def center_matrix(original_matrix, target_width=3200):
    
    original_height, original_width = original_matrix.shape  
    
    new_matrix = np.zeros((50, target_width))

    if original_width <= target_width:
        start_index = (target_width - original_width) // 2
        # Place the original matrix in the center of the new matrix
        new_matrix[:, start_index:start_index + original_width] = original_matrix
    else:
        # Crop the original matrix to fit into the target width
        start_crop_index = (original_width - target_width) // 2
        cropped_matrix = original_matrix[:, start_crop_index:start_crop_index + target_width]
        new_matrix = cropped_matrix
    return new_matrix


class SpectrogramDataloader(Dataset):
    def __init__(self, labels, wav_file_path):
        self.labels = labels  
        self.wav_file_path = wav_file_path
        
        # Read the .wav file to get the sampling frequency
        self.fs, _ = wavfile.read(wav_file_path)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Read the waveform directly from the .wav file at each call
        self.fs, waveform = wavfile.read(self.wav_file_path)

        waveform = np.asarray(waveform)

        # Compute the STFT using the extracted sampling frequency
        f, t_stft, Zxx = stft(waveform, fs=self.fs, nperseg=100)

        one_spectrogram = Zxx
        
        # Normalisation
        one_spectrogram = (one_spectrogram - one_spectrogram.min()) / (one_spectrogram.max() - one_spectrogram.min())
        
        # Reshape to add channel dimension
        one_spectrogram = np.expand_dims(one_spectrogram, axis=0)  
        
        one_label = self.labels[idx]
        one_spectrogram = one_spectrogram.astype(np.float32)
        
        return torch.tensor(one_spectrogram), torch.tensor(one_label).float()
    

class CNNFull(pl.LightningModule):
    def __init__(self, n_classes, window_size, stride):
        super(CNNFull, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=N_CHANNELS, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Calculate input size for fully connected layer
        self.fc1_input_size = 512000  # Update this based on your input size calculation
        self.fc1 = nn.Linear(self.fc1_input_size, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, n_classes)

        # Parameters for sliding window
        self.window_size = window_size  # In seconds
        self.stride = stride  # In seconds

    def forward(self, x):
        x = abs(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)  # Flatten the tensor
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return torch.sigmoid(x)

    def sliding_window_inference(self, wav_file_path):
        # Load the audio file
        fs, waveform = wavfile.read(wav_file_path)
        
        # Convert window size and stride from seconds to samples
        window_samples = int(self.window_size * fs)
        stride_samples = int(self.stride * fs)
        
        # Number of sliding windows
        num_windows = (len(waveform) - window_samples) // stride_samples + 1

        predictions = []

        for i in range(num_windows):
            # Extract the segment for the current window
            start_idx = i * stride_samples
            end_idx = start_idx + window_samples
            window_waveform = waveform[start_idx:end_idx]

            # Compute STFT to get the spectrogram
            _, _, Zxx = stft(window_waveform, fs=fs, nperseg=100)

            Zxx = Zxx[:50,]  # Limit frequency resolution to match CNN input
            Zxx = center_matrix(Zxx, target_width=3200)  # Ensure target size for CNN input
            
            # Normalization
            Zxx = (Zxx - Zxx.min()) / (Zxx.max() - Zxx.min())
            Zxx = np.expand_dims(Zxx, axis=0)  # Add channel dimension
            
            # Convert to tensor
            input_tensor = torch.tensor(Zxx).float().unsqueeze(0)  # Add batch dimension

            # Forward pass through the CNN
            pred = self.forward(input_tensor)

            # Store the prediction
            predictions.append(pred.item())

        return np.array(predictions)
    
    def filter_drums_from_audio(self, wav_file_path, output_path):
        # Load the audio file
        fs, waveform = wavfile.read(wav_file_path)

        # Convert window size and stride from seconds to samples
        window_samples = int(self.window_size * fs)
        stride_samples = int(self.stride * fs)

        # Number of sliding windows
        num_windows = (len(waveform) - window_samples) // stride_samples + 1

        filtered_waveform = np.zeros_like(waveform)  # Initialize an empty filtered waveform
        overlap_counts = np.zeros_like(waveform)  # To manage overlap

        for i in range(num_windows):
            # Extract the segment for the current window
            start_idx = i * stride_samples
            end_idx = start_idx + window_samples
            window_waveform = waveform[start_idx:end_idx]

            # Compute STFT to get the spectrogram
            f, t, Zxx = stft(window_waveform, fs=fs, nperseg=100)

            # Normalize and prepare input for CNN
            Zxx_mag = np.abs(Zxx)
            Zxx_mag_norm = (Zxx_mag - Zxx_mag.min()) / (Zxx_mag.max() - Zxx_mag.min())
            Zxx_mag_norm = np.expand_dims(Zxx_mag_norm[:50,], axis=0)  # Add channel dimension
            
            input_tensor = torch.tensor(Zxx_mag_norm).float().unsqueeze(0)  # Add batch dimension

            # Get prediction from the model
            pred = self.forward(input_tensor).item()

            if pred > 0.5:  # If the model predicts drums in this window
                # Apply a frequency filter to the drum regions in the spectrogram
                Zxx_filtered = self.apply_drum_filter(Zxx, f)
            else:
                Zxx_filtered = Zxx  # No modification for non-drum windows

            # Inverse STFT to convert back to time-domain signal
            _, filtered_segment = istft(Zxx_filtered, fs=fs)

            # Add the filtered segment back to the waveform
            filtered_waveform[start_idx:end_idx] += filtered_segment[:window_samples]
            overlap_counts[start_idx:end_idx] += 1  # Keep track of overlaps

        # Normalize the filtered waveform in the regions where there was overlap
        filtered_waveform /= np.maximum(overlap_counts, 1)

        # Save the filtered waveform to an output file
        wavfile.write(output_path, fs, filtered_waveform.astype(np.int16))

    def apply_drum_filter(self, Zxx, f):
        """
        Apply a drum filter to the STFT spectrogram Zxx by attenuating specific frequencies.
        :param Zxx: STFT spectrogram (complex-valued).
        :param f: Frequency bins from the STFT.
        :return: Filtered STFT spectrogram.
        """
        # Example: Notch filter to attenuate low frequencies (e.g., < 200 Hz)
        f_threshold = 200  # This threshold can be tuned based on drum characteristics
        for i, freq in enumerate(f):
            if freq < f_threshold:
                Zxx[i, :] *= 0.2  # Attenuate the low frequencies (e.g., drums)
        return Zxx