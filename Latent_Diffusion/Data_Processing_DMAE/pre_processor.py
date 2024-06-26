import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.nn import ConstantPad1d
from tqdm import tqdm

from .Chunk_Dataset import AudioChunkDataSet


class PreProcessor:
    """
    Class that handles the pre-processing for the training involving the encoder-decoder model.
    """

    def __init__(
        self,
        input_audio_dir: str, 
        chunked_dir: str,
        meta_data_callback: callable = None,
        desired_sample_rate: int = 44100,
        input_len: int = 2**18,
        checkpoint_size: int = 50,
        delete_source: bool = False,
    ) -> None:
        self.meta_data_callback: callable = meta_data_callback
        self.meta_data: pd.DataFrame = None
        self.input_dir = input_audio_dir
        self.checkpoint_size = checkpoint_size
        self.chunked_dir = chunked_dir
        self.sample_rate = desired_sample_rate
        self.input_tensor_len = input_len
        self.metadata_file_path = os.path.join(self.chunked_dir, "metadata.csv")
        self.output_dir = os.path.join(self.chunked_dir, "waveforms")
        self.training_metadata_dest = os.path.join(
            self.chunked_dir, "train_metadata.csv"
        )
        self.val_metadata_dest = os.path.join(self.chunked_dir, "val_metadata.csv")
        self.delete_source = delete_source
        # Create any directories that will be needed later
        dirs_to_create = [
            self.input_dir,
            self.output_dir,
            self.chunked_dir,
        ]
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)

    def chunk_single_song(self, audio_file: str):
        """
        Chunking Procedure for a single audio file.

        Note: You will need enough memory to perform the forward pass for the encoder model.

        Be mindful of this when encoding.
        """
        song_path = os.path.join(self.input_dir, audio_file)
        splitted = self.split_audio(song_path)

        song_name = audio_file.removesuffix(
            ".wav"
        )  # Considering Song name as file name without file extension.
        # Creating a specific directory for song name.
        os.makedirs(os.path.join(self.output_dir, song_name), exist_ok=True)

        # Adding Metadata
        temp_df = pd.DataFrame(
            [
                self.naming_convention(
                    song_name, start_time=i + 1, total_chunks=splitted.shape[0]
                )
                for i in range(splitted.shape[0])
            ],
            columns=["Chunk"],
        )
        temp_df["Song_Name"] = song_name

        # If Additional Song-level meta data needs to be added then implement callback.
        if self.meta_data_callback:
            temp_df = self.meta_data_callback(audio_file, temp_df)

        # Accumulating Latent Tensor
        for i in range(splitted.shape[0]):
            music_path = os.path.join(
                self.output_dir,
                self.naming_convention(song_name, i + 1, splitted.shape[0]) + ".wav",
            )
            torchaudio.save(
                music_path,
                src=splitted[i, :, :].squeeze(),
                format="wav",
                sample_rate=self.sample_rate,
            )

        if self.delete_source:
            os.remove(song_path)

        return temp_df

    def preprocess(self) -> None:
        """
        Chunk dataset into output directory.
        """
        # Making Directories if these don't already exist
        if not os.path.exists(self.input_dir) or len(os.listdir(self.input_dir)) == 0:
            raise EnvironmentError(
                f"Please download the dataset and place in {self.input_dir} folder"
            )

        audio_files = os.listdir(self.input_dir)
        # Ensure all files are *.wav
        # If a song's download was interrupted, it may have a file name like song.wav.axszd
        broken_file_list = [file for file in audio_files if not file.endswith(".wav")]
        if len(broken_file_list):
            for file in broken_file_list:
                os.remove(os.path.join(self.input_dir, file))
            raise EnvironmentError(
                "Deleting The following songs that were incorrectly downloaded and exiting: ", broken_file_list
            )

        output_df = pd.DataFrame()

        for audio_file in tqdm(audio_files):
            try:
                temp_df = self.chunk_single_song(audio_file)
            except Exception as e:
                print("Caught exception:" + str(e))
                continue

            output_df = pd.concat([output_df, temp_df])

        self.meta_data = output_df
        print("--- Saving Meta Data DataFrame ---")
        output_df.to_csv(self.metadata_file_path, index=False)
        print(
            "Number of successfully chunked songs --"
            + str(len(os.listdir(self.output_dir)) - 1)
        )

    def split_audio(self, audio_file_path, pad_element=0):
        """
        Splits and pads audio based on desired tensor_len
        """
        audio_tensor, sample_rate = torchaudio.load(audio_file_path)
        if self.sample_rate != sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            audio_tensor = resampler(audio_tensor)
        splitted = list(
            torch.split(
                audio_tensor, split_size_or_sections=self.input_tensor_len, dim=1
            )
        )
        last_splitted = splitted[-1]
        splitted[-1] = ConstantPad1d(
            (0, self.input_tensor_len - last_splitted.shape[1]), pad_element
        )(last_splitted)
        return torch.stack(splitted, dim=0)

    def naming_convention(self, audio_file, start_time, total_chunks):
        """
        Function that does the naming convention for the splits
        """
        return os.path.join(audio_file, f"chunk {start_time} of {total_chunks}")

    def update_meta_data(self):
        """
        make a call to the meta data directory and store in class variable.
        """
        try:
            self.meta_data = pd.read_csv(self.metadata_file_path)
        except:
            raise Exception("No Metadata or incorrectly specified metadata path")
        return self.meta_data

    def return_dataset(self, meta_data: pd.DataFrame = None):
        """
        Instantiate and return an instance of torch's dataloader class consistent with
        the metadata csv, and the directories provided in the constructor
        """
        if len(os.listdir(self.output_dir)) == 0:
            raise Exception("No Chunked Audio Files -- Check audio directories")

        # Implementing get_audio_sample from meta_data method
        def _get_audio_sample_path_from_meta_data(index):
            """
            Function specifying how to get the index from meta_data attribute.
            """
            row = self.meta_data.iloc[index]
            return os.path.join(self.output_dir, row["Chunk"]) + ".wav"

        if meta_data is None:
            meta_data = self.update_meta_data()
        return AudioChunkDataSet(
            meta_data,
            self.output_dir,
            _get_audio_sample_path_from_meta_data,
            prompt_constructor=None,
        )

    def construct_meta_data_file(self):
        """
        Based on the output directory provided in the constructor,
        creates a new metadata.csv, updates the class variable and returns the file.
        """
        return_df = pd.DataFrame()

        filenames = os.listdir(self.output_dir)
        for file in tqdm(filenames, desc="Constructing MetaData Dataframe"):
            num_chunks = len(os.listdir(os.path.join(self.output_dir, file)))
            # Adding Metadata
            temp_df = pd.DataFrame(
                [
                    self.naming_convention(
                        file, start_time=i + 1, total_chunks=num_chunks
                    )
                    for i in range(num_chunks)
                ],
                columns=["Chunk"],
            )
            temp_df["Song_Name"] = file
            return_df = pd.concat([return_df, temp_df])
        self.meta_data = return_df
        self.meta_data.to_csv(self.metadata_file_path)

    def construct_train_split_data_files(self, train_prop=0.95):
        """
        Construct meta data file then randomly splits metadata into
        train and test Audio Chunk dataset classes.
        """
        self.construct_meta_data_file()
        return self.split_into_train_val(train_prop=train_prop)

    def split_into_train_val(self, train_prop=0.95):
        """
        Splitting the metadata file into train and test and returning torch datasets accordingly.
        Note: This function splits by songs, not by number of samples. (so the proportions will be approximate).
        """
        self.update_meta_data()
        songs_array = self.meta_data["Song_Name"].unique()
        np.random.shuffle(songs_array)
        songs_list = songs_array.tolist()
        train_songs = songs_list[: int(np.floor(train_prop * len(songs_list)))]
        val_songs = songs_list[int(np.floor(train_prop * len(songs_list))) :]

        # if len(train_songs)== 0 or len(val_songs) == 0:
        #    raise Exception('Validation set or train set empty -- specify a proportion that allows for model validation and training.')
        if len(train_songs) == 1:
            train_songs = pd.DataFrame([train_songs], columns=["Song_Name"])
        else:
            train_songs = pd.DataFrame(train_songs, columns=["Song_Name"])

        if len(val_songs) == 1:
            val_songs = pd.DataFrame([val_songs], columns=["Song_Name"])
        else:
            val_songs = pd.DataFrame(val_songs, columns=["Song_Name"])

        train_meta_data = pd.merge(self.meta_data, train_songs, on="Song_Name")
        val_meta_data = pd.merge(self.meta_data, val_songs, on="Song_Name")

        train_meta_data.to_csv(self.training_metadata_dest)
        val_meta_data.to_csv(self.val_metadata_dest)

        return self.return_dataset(train_meta_data), self.return_dataset(val_meta_data)

    def retrieve_train_test_split(self):
        """
        Getting train-test split if it already exists in the data folder.
        Otherwise throw exception saying to call the split_into_train_val set.
        """
        train_meta_data = pd.read_csv(self.training_metadata_dest)
        val_meta_data = pd.read_csv(self.val_metadata_dest)
        return self.return_dataset(train_meta_data), self.return_dataset(val_meta_data)
