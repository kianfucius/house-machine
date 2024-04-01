import pandas as pd
import torchaudio
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
        prompt_constructor: callable,
    ) -> None:
        super().__init__()
        self.prompt_constructor = prompt_constructor
        self.audio_dir = preprocessed_audio_dir
        self._get_audio_sample_path = get_audio_sample_path_func
        self.meta_data = meta_data

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        signal,sample_rate = torchaudio.load(audio_sample_path)
        return signal

