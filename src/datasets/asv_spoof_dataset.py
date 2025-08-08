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
        **kwargs,
    ):
        protocol_path = os.path.join(
            la_root, "ASVspoof2019_LA_cm_protocols", protocol_file
        )
        audio_base_path = os.path.join(la_root, audio_folder)

        index = self._make_index(protocol_path, audio_base_path)

        self.use_stft = use_stft
        self.stft_params = stft_params or {}

        if self.use_stft:
            self.window = torch.blackman_window(self.stft_params["win_length"])

        super().__init__(index=index, **kwargs)

    def load_object(self, path):
        waveform, _ = torchaudio.load(path)

        if self.use_stft:
            stft_output = torch.stft(
                waveform,
                n_fft=self.stft_params["n_fft"],
                hop_length=self.stft_params["hop_length"],
                win_length=self.stft_params["win_length"],
                window=self.window,
                return_complex=self.stft_params.get("return_complex", True),
            )

            magnitude = stft_output.abs()
        else:
            magnitude = waveform

        magnitude = magnitude[..., :600]
        magnitude = self._pad_or_crop_time(magnitude, target_time=600)
        return magnitude

    def _make_index(self, protocol_path, audio_base_path):
        index = []
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                file_id = parts[1]
                label_str = parts[-1].lower()
                label = 1 if label_str == "bonafide" else 0
                audio_path = os.path.join(audio_base_path, file_id + ".flac")
                index.append(
                    {
                        "path": audio_path,
                        "label": label,
                        "id": file_id,
                    }
                )

        return index

    def _pad_or_crop_time(self, tensor, target_time=600):
        c, f, t = tensor.shape

        output = torch.zeros(
            (c, f, target_time), dtype=tensor.dtype, device=tensor.device
        )
        length = min(t, target_time)
        output[:, :, :length] = tensor[:, :, :length]
        return output
