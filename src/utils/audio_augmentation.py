import numpy as np
import librosa
from scipy.signal import convolve
import random

class AudioAugmentor:
    def __init__(self, config):
        """
        Initialize the AudioAugmentor with a configuration.
        :param config: Dictionary containing augmentation probabilities and parameter ranges.
        """
        self.config = config
        self.augmentations = {
            "time_stretch": self.time_stretch,
            "pitch_shift": self.pitch_shift,
            "add_noise": self.add_noise,
            "dynamic_range_compression": self.dynamic_range_compression,
            "add_reverberation": self.add_reverberation,
            "crop_audio": self.crop_audio,
            "pad_audio": self.pad_audio,
        }

    def augment(self, audio, sr):
        """
        Apply a sequence of augmentations to the audio.
        :param audio: Input audio waveform.
        :param sr: Sampling rate.
        :return: Augmented audio waveform.
        """
        for aug_name, aug_func in self.augmentations.items():
            if random.random() < self.config.get(aug_name, {}).get("probability", 0):
                params = self.config[aug_name]
                audio = aug_func(audio, sr, **params)
        return audio

    def augment_batch(self, batch, sr):
        """
        Apply augmentations to a batch of audio waveforms.
        :param batch: List or NumPy array of audio waveforms.
        :param sr: Sampling rate.
        :return: List of augmented audio waveforms.
        """
        return [self.augment(audio, sr) for audio in batch]

    def time_stretch(self, audio, sr, rate_range):
        rate = random.uniform(*rate_range)
        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(self, audio, sr, n_steps_range):
        n_steps = random.uniform(*n_steps_range)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    def add_noise(self, audio, sr, noise_level_range):
        noise_level = random.uniform(*noise_level_range)
        noise = np.random.normal(0, noise_level, audio.shape)
        return audio + noise

    def dynamic_range_compression(self, audio, sr, factor_range):
        factor = random.uniform(*factor_range)
        return np.sign(audio) * (np.abs(audio) ** factor)

    def add_reverberation(self, audio, sr, reverb_factor_range):
        reverb_factor = random.uniform(*reverb_factor_range)
        impulse_response = np.zeros(int(sr * reverb_factor))
        impulse_response[::int(sr * reverb_factor / 10)] = 1
        impulse_response /= np.sum(impulse_response)  # Normalize the impulse response
        return convolve(audio, impulse_response, mode='full')[:len(audio)]

    def crop_audio(self, audio, sr, crop_duration_range):
        crop_duration = random.uniform(*crop_duration_range)
        crop_samples = int(crop_duration * sr)
        return audio[:crop_samples] if crop_samples < len(audio) else audio

    def pad_audio(self, audio, sr, target_duration):
        target_samples = int(target_duration * sr)
        if len(audio) < target_samples:
            padding = target_samples - len(audio)
            return np.pad(audio, (0, padding), mode='constant')
        return audio