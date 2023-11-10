import logging
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

import constants

from .Chunk_Dataset import AudioChunkDataSet


class PreProcessor:
    """
    Class that handles the pre-processing for the training involving the encoder-decoder model.
    """

    def __init__(
        self,
        input_audio_dir=constants.RAW_MP3_DIR,
        output_dir=constants.ENCODER_DECODER_PROCESSED_DIR,
        desired_sample_rate=48000,
        tensor_len=2**18,
    ) -> None:
        self.meta_data = None
        self.input_dir = input_audio_dir
        self.output_dir = output_dir
        self.sample_rate = desired_sample_rate
        self.tensor_len = tensor_len
        self.metadata_dir = os.path.join(self.output_dir, "metadata") + ".csv"
        self.audio_dir = os.path.join(self.output_dir, "waveforms")
        self.training_meta_dir = (
            os.path.join(self.output_dir, "train_metadata") + ".csv"
        )
        self.val_meta_dir = os.path.join(self.output_dir, "val_metadata") + ".csv"
        # Making Directories if these don't already exist
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)

    def chunk_single_song(self, audio_file):
        """
        Chunking Procedure for a single audio file.
        """
        splitted, sample_rate, start_times, time_per_split = self.split_audio(
            os.path.join(self.input_dir, audio_file)
        )
        song_name = audio_file[:-4]
        temp_df = pd.DataFrame(start_times, columns=["start_time"])
        temp_df["sample_rate"] = sample_rate
        temp_df["time_len"] = time_per_split
        temp_df["song_name"] = song_name
        parent_dir = os.path.join(self.audio_dir, song_name)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        for i in range(len(splitted)):
            music_path = os.path.join(parent_dir, str(int(start_times[i]))) + ".wav"
            torchaudio.save(
                music_path,
                splitted[i],
                sample_rate=self.sample_rate,
                channels_first=True,
            )
        return temp_df

    def preprocess(self, verbose=False) -> None:
        """
        Chunk dataset into output directory.
        """
        audio_files_list = os.listdir(self.input_dir)
        with mp.Pool(mp.cpu_count() - 2) as p:
            return_meta_list = list(
                tqdm(
                    p.imap(self.chunk_single_song, audio_files_list),
                    total=len(audio_files_list),
                )
            )
        metadata_df = pd.concat(return_meta_list)
        metadata_df.to_csv(os.path.join(self.metadata_dir), index=False)
        self.meta_data = metadata_df

    def split_audio(self, audio_file_path, verbose=False):
        """
        This function splits audio based on desired tensor_len
        """
        audio_tensor, sample_rate = torchaudio.load(audio_file_path)
        if self.sample_rate != sample_rate and verbose:
            logging.info(
                "Sample rate not desired in file with path: " + audio_file_path
            )
        splitted = torch.split(
            audio_tensor, split_size_or_sections=self.tensor_len, dim=1
        )[:-1]
        time_per_split = (audio_tensor.shape[1] / sample_rate) / len(splitted)
        start_times = (
            torch.arange(start=0, end=len(splitted)) * time_per_split
        ).tolist()
        return splitted, sample_rate, start_times, time_per_split

    def naming_convention(self, audio_file, start_time):
        """
        Function that does the naming convention for the splits
        """
        return os.path.join(audio_file, str(start_time))

    def update_meta_data(self):
        """
        make a call to the meta data directory and store in class variable.
        """
        try:
            self.meta_data = pd.read_csv(self.metadata_dir)
        except:
            raise Exception("No Metadata or incorrectly specified metadata path")
        return self.meta_data

    def return_dataset(
        self, meta_data: pd.DataFrame = None, additional_meta_data: list = None
    ):
        """
        Instantiate and return an instance of torch's dataloader class consistent with
        the metadata csv, and the directories provided in the constructor
        """
        if len(os.listdir(self.audio_dir)) == 0:
            raise Exception("No Chunked Audio Files -- Check audio directories")

        # Implementing get_audio_sample from meta_data method
        def _get_audio_sample_path_from_meta_data(index):
            """
            Function specifying how to get the index from meta_data attribute.
            """
            row = self.meta_data.iloc[index]
            return (
                os.path.join(
                    self.audio_dir,
                    self.naming_convention(row["song_name"], int(row["start_time"])),
                )
                + ".wav"
            )

        if meta_data is None:
            meta_data = self.update_meta_data()

        meta_data_list = ["song_name", "start_time"]
        if additional_meta_data:
            meta_data_list = meta_data_list + additional_meta_data
            return AudioChunkDataSet(
                meta_data[meta_data_list],
                self.audio_dir,
                _get_audio_sample_path_from_meta_data,
                get_metadata=meta_data_list,
            )
        else:
            return AudioChunkDataSet(
                meta_data, self.audio_dir, _get_audio_sample_path_from_meta_data
            )

    def split_into_train_val(self, train_prop=0.95, additional_meta_data: list = None):
        """
        Splitting the metadata file into train and test and returning torch datasets accordingly.
        Note: This function splits by songs, not by number of samples. (so the proportions will be approximate).
        """
        self.update_meta_data()
        songs_array = self.meta_data["song_name"].unique()
        np.random.shuffle(songs_array)
        songs_list = songs_array.tolist()
        train_songs = songs_list[: int(np.floor(train_prop * len(songs_list)))]
        val_songs = songs_list[int(np.floor(train_prop * len(songs_list))) :]

        # if len(train_songs)== 0 or len(val_songs) == 0:
        #    raise Exception('Validation set or train set empty -- specify a proportion that allows for model validation and training.')
        if len(train_songs) == 1:
            train_songs = pd.DataFrame([train_songs], columns=["song_name"])
        else:
            train_songs = pd.DataFrame(train_songs, columns=["song_name"])

        if len(val_songs) == 1:
            val_songs = pd.DataFrame([val_songs], columns=["song_name"])
        else:
            val_songs = pd.DataFrame(val_songs, columns=["song_name"])

        train_meta_data = pd.merge(self.meta_data, train_songs, on="song_name")
        val_meta_data = pd.merge(self.meta_data, val_songs, on="song_name")

        train_meta_data.to_csv(self.training_meta_dir)
        val_meta_data.to_csv(self.val_meta_dir)

        return self.return_dataset(
            train_meta_data, additional_meta_data
        ), self.return_dataset(val_meta_data, additional_meta_data)

    def retrieve_train_test_split(self):
        """
        Getting train-test split if it already exists in the data folder.
        Otherwise throw exception saying to call the split_into_train_val set.
        """
        train_meta_data = pd.read_csv(self.training_meta_dir)
        val_meta_data = pd.read_csv(self.val_meta_dir)
        return self.return_dataset(train_meta_data), self.return_dataset(val_meta_data)
