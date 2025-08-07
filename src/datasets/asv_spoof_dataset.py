import os

import torch
import torchaudio

from src.datasets.base_dataset import BaseDataset


class ASVspoofDataset(BaseDataset):
    def __init__(
        self,
        la_root,
        protocol_file,
        audio_folder,
        use_stft=False,
        stft_params=None,
        **kwargs
    ):
        protocol_path = os.path.join(
            la_root, "ASVspoof2019_LA_cm_protocols", protocol_file
        )
        audio_base_path = os.path.join(la_root, audio_folder)

        index = self._make_index(protocol_path, audio_base_path)

        self.use_stft = use_stft
        self.stft_params = stft_params

        if self.use_stft:
            self.window = torch.hamming_window(stft_params["win_length"])

        super().__init__(index=index, **kwargs)

    def load_object(self, path):
        waveform, sr = torchaudio.load(path)

        if self.use_stft:
            window = torch.hamming_window(self.stft_params["win_length"])
            stft_complex = torch.stft(
                waveform,
                n_fft=self.stft_params["n_fft"],
                hop_length=self.stft_params["hop_length"],
                win_length=self.stft_params["win_length"],
                window=window,
                return_complex=True,
            )
            magnitude_squared = stft_complex.abs() ** self.stft_params.get("power", 2)
            return magnitude_squared

        return waveform

    def _make_index(self, protocol_path, audio_base_path):
        index = []
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                file_id = parts[0]
                label_str = parts[-1].lower()
                label = 1 if label_str == "bonafide" else 0
                audio_path = os.path.join(audio_base_path, file_id + ".flac")
                index.append(
                    {
                        "path": audio_path,
                        "label": label,
                    }
                )

        return index
