## librosa
import numpy as np

def librosa_time_to_samples(times, sr=22050):
    return (np.asanyarray(times) * sr).astype(int)

def librosa_samples_to_frames(samples, hop_length=512, n_fft=None):
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)
    samples = np.asanyarray(samples)
    return np.floor((samples - offset) // hop_length).astype(int)


def librosa_time_to_frames(times, sr=22050, hop_length=512, n_fft=None):
    samples = librosa_time_to_samples(times, sr=sr)
    return librosa_samples_to_frames(samples, hop_length=hop_length, n_fft=n_fft)

def librosa_frames_to_samples(frames, hop_length=512, n_fft=None):
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)

def librosa_samples_to_time(samples, sr=22050):
    return np.asanyarray(samples) / float(sr)


def librosa_frames_to_time(frames, sr=22050, hop_length=512, n_fft=None):
    samples = librosa_frames_to_samples(frames,
                                hop_length=hop_length,
                                n_fft=n_fft)
    return librosa_samples_to_time(samples, sr=sr)


