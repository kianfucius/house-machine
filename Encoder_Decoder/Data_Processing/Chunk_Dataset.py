import pandas as pd
import torch
from torch.utils.data import Dataset


class AudioChunkDataSet(Dataset):
    """
    Torch Dataset Class which will be used to load the chunked audio into the encoder-decoder model.
    """

    def __init__(
        self,
        meta_data: pd.DataFrame,
        preprocessed_audio_dir: str,
        get_audio_sample_path_func: callable,
        get_metadata: list = None,
    ) -> None:
        super().__init__()
        self.get_metadata = get_metadata
        if self.get_metadata:
            self.only_meta_df = meta_data[get_metadata]
        self.audio_dir = preprocessed_audio_dir
        self._get_audio_sample_path = get_audio_sample_path_func
        self.meta_data = meta_data

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        signal, sr = torch.load(audio_sample_path)
        if self.get_metadata:
            return signal, self.only_meta_df.iloc[index].to_dict()
        else:
            return signal
