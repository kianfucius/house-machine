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
        signal, sample_rate = torchaudio.load(audio_sample_path)
        # A bunch of validation checks, but this realistically shouldn't throw any errors
        if sample_rate != 44100:
            raise ValueError(f"Sample rate is not 44100: {audio_sample_path}")
        if signal.shape[0] != 2:
            raise ValueError(f"Signal is not stereo: {audio_sample_path}")
        if signal.shape[1] < 262144:
            raise ValueError(
                f"Signal {audio_sample_path} was not padded correctly: {signal.shape[1]}"
            )
        return signal
