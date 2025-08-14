from functools import partial

import librosa
import numpy as np
import scipy


class Sequential:
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, inp: np.ndarray):
        res = inp
        for transform in self.transforms:
            res = transform(res)
        return res


class Windowing:
    def __init__(self, window_size=1024, hop_length=None):
        self.window_size = window_size
        self.hop_length = hop_length if hop_length else self.window_size // 2
    
    def __call__(self, waveform):
        # Your code here
        # raise NotImplementedError("TODO: assignment")
        wav_length = waveform.shape[0]
        
        left_padding = np.zeros(self.window_size // 2)
        right_padding = np.zeros(self.window_size // 2)
        
        new_waveform = np.concatenate((left_padding, waveform, right_padding)) # wav_length + window_size - (window_size % 2)
        
        windows = []
        for idx in range((wav_length - self.window_size % 2) // self.hop_length + 1):
            left_idx = idx * self.hop_length
            right_idx = left_idx + self.window_size
            windows.append(new_waveform[left_idx : right_idx])
        
        windows = np.stack(windows)
        return windows
        # ^^^^^^^^^^^^^^
        
        return windows
    

class Hann:
    def __init__(self, window_size=1024):
        # Your code here
        self.hann_weights = scipy.signal.windows.hann(window_size, sym=False) # window_size
        # ^^^^^^^^^^^^^^

    
    def __call__(self, windows):
        # Your code here
        return windows * self.hann_weights
        # ^^^^^^^^^^^^^^



class DFT:
    def __init__(self, n_freqs=None):
        self.n_freqs = n_freqs

    def __call__(self, windows): # (T x window_size) = (T x N)
        # Your code here
        freqs = np.fft.rfft(windows) # (T x N // 2), complex
        freqs = np.absolute(freqs)
        if self.n_freqs is not None:
            freqs = freqs[:, :self.n_freqs]
        # ^^^^^^^^^^^^^^

        return freqs


class Square:
    def __call__(self, array):
        return np.square(array)


class Mel:
    def __init__(self, n_fft, n_mels=80, sample_rate=22050):
        # Your code here
        self.mel_matrix = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=1, fmax=8192, htk=False)
        self.inv_mel_matrix = np.linalg.pinv(self.mel_matrix)
        # ^^^^^^^^^^^^^^


    def __call__(self, spec):
        # Your code here
        return spec @ self.mel_matrix.T
        # ^^^^^^^^^^^^^^

    def restore(self, mel):
        # Your code here
        return mel @ self.inv_mel_matrix.T
        # ^^^^^^^^^^^^^^


class GriffinLim:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.griffin_lim = partial(
            librosa.griffinlim,
            n_iter=32,
            hop_length=hop_length,
            win_length=window_size,
            n_fft=window_size,
            window='hann'
        )

    def __call__(self, spec):
        return self.griffin_lim(spec.T)


class Wav2Spectrogram:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.windowing = Windowing(window_size=window_size, hop_length=hop_length)
        self.hann = Hann(window_size=window_size)
        self.fft = DFT(n_freqs=n_freqs)
        # self.square = Square()
        self.griffin_lim = GriffinLim(window_size=window_size, hop_length=hop_length, n_freqs=n_freqs)

    def __call__(self, waveform):
        return self.fft(self.hann(self.windowing(waveform)))

    def restore(self, spec):
        return self.griffin_lim(spec)


class Wav2Mel:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None, n_mels=80, sample_rate=22050):
        self.wav_to_spec = Wav2Spectrogram(
            window_size=window_size,
            hop_length=hop_length,
            n_freqs=n_freqs)
        self.spec_to_mel = Mel(
            n_fft=window_size,
            n_mels=n_mels,
            sample_rate=sample_rate)

    def __call__(self, waveform):
        return self.spec_to_mel(self.wav_to_spec(waveform))

    def restore(self, mel):
        return self.wav_to_spec.restore(self.spec_to_mel.restore(mel))


class TimeReverse:
    def __call__(self, mel):
        # Your code here
        reverse_mel = mel[::-1, :]
        return reverse_mel
        # ^^^^^^^^^^^^^^



class Loudness:
    def __init__(self, loudness_factor):
        # Your code here
        self.loudness_factor = loudness_factor
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        return mel * self.loudness_factor
        # ^^^^^^^^^^^^^^




class PitchUp:
    def __init__(self, num_mels_up):
        # Your code here
        self.num_mels_up = num_mels_up
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        time_frames, num_mels = mel.shape
        up_mel = np.zeros_like(mel)
        if self.num_mels_up < num_mels:
            up_mel[:, self.num_mels_up :] = mel[:, : num_mels - self.num_mels_up]
        return up_mel
        # ^^^^^^^^^^^^^^



class PitchDown:
    def __init__(self, num_mels_down):
        # Your code here
        self.num_mels_down = num_mels_down
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        time_frames, num_mels = mel.shape
        down_mel = np.zeros_like(mel)
        if self.num_mels_down < num_mels:
            down_mel[:, : num_mels - self.num_mels_down] = mel[:, self.num_mels_down:]
        return down_mel
        # ^^^^^^^^^^^^^^



class SpeedUpDown:
    def __init__(self, speed_up_factor=1.0):
        # Your code here
        self.speed_up_factor = speed_up_factor
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        indices = np.linspace(0, mel.shape[0] - 1, int(self.speed_up_factor * mel.shape[0]))
        indices = np.round(indices).astype("int")
        return mel[indices, :]
        # ^^^^^^^^^^^^^^



class FrequenciesSwap:
    def __call__(self, mel):
        # Your code here
        reverse_mel = mel[:, ::-1]
        return reverse_mel
        # ^^^^^^^^^^^^^^



class WeakFrequenciesRemoval:
    def __init__(self, quantile=0.05):
        # Your code here
        self.quantile = quantile
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        threshold = np.quantile(mel, self.quantile)
        return np.where(mel >= threshold, mel, 0.)
        # ^^^^^^^^^^^^^^



class Cringe1:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class Cringe2:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^

