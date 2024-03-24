import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from audiotools import AudioSignal
from torch.nn import ConstantPad1d
from tqdm import tqdm
from multiprocessing import Pool

from .Chunk_Dataset import AudioChunkDataSet


class PreProcessor:
    """
    Class that handles the pre-processing for the training involving the encoder-decoder model.
    """

    def __init__(
        self,
        meta_data_callback: callable = None,
        input_audio_dir="Latent_Diffusion/data",
        chunked_dir="Latent_Diffusion/chunked_data",
        desired_sample_rate=44100,
        input_len=2**18,
        checkpoint_size=50,
    ) -> None:
        self.meta_data_callback = meta_data_callback

        self.meta_data = None
        self.input_dir = input_audio_dir
        self.checkpoint_size = checkpoint_size
        self.chunked_dir = chunked_dir
        self.sample_rate = desired_sample_rate
        self.input_tensor_len = input_len
        self.metadata_file_path = os.path.join(self.chunked_dir, "metadata.csv")
        self.output_dir = os.path.join(self.chunked_dir, "waveforms")
        self.training_meta_dir = os.path.join(self.chunked_dir, "train_metadata.csv")
        self.val_meta_dir = os.path.join(self.chunked_dir, "val_metadata.csv")

    def chunk_single_song(self, audio_file):
        """
        Chunking Procedure for a single audio file.

        Note: You will need enough memory to perform the forward pass for the encoder model.

        Be mindful of this when encoding.
        """
        splitted = self.split_audio(os.path.join(self.input_dir, audio_file))

        song_name = audio_file.removesuffix(
            ".wav"
        )  # Considering Song name as file name without file extension.
        # Creating a specific directory for song name.
        parent_dir = os.path.join(self.output_dir, song_name)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

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
                src = splitted[i, :, :].squeeze(),
                format = 'wav',
                sample_rate = self.sample_rate)
                

        # Removing from initial directoy:
        remove_path = os.path.join(self.input_dir, song_name) + '.wav'
        if os.path.isfile(remove_path):
            os.remove(remove_path)
        else:
            # If it fails, inform the user.
            print("Error: %s file not found" % song_name)

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
        
        audio_files_list = os.listdir(self.input_dir)
        output_df = pd.DataFrame()

        audio_files_list = audio_files_list
        
        # with Pool(4) as pool:
        #     for index, result_df in enumerate(
        #         tqdm(
        #             pool.imap_unordered(self.chunk_single_song, audio_files_list),
        #             total=len(audio_files_list),
        #         )
        #     ):
        #         output_df = pd.concat([output_df, result_df])

        for audio_file in tqdm(audio_files_list):
            try:
                temp_df = self.chunk_single_song(audio_file)
            except Exception as e:
                print('Caught exception:' + str(e))
                continue
            
            output_df = pd.concat([output_df, temp_df])

        self.meta_data = output_df
        print('--- Saving Meta Data DataFrame ---')
        output_df.to_csv(self.metadata_file_path, index=False)
        print('Number of successfully chunked songs --' + str(len(os.listdir(self.output_dir)) -1 ))

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
        for file in filenames:
            num_chunks = len(os.listdir(os.path.join(self.output_dir)))    
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
        return return_df
            
    def construct_train_split_data_files(self, train_prop = 0.95):
        """
        Construct meta data file then randomly splits metadata into
        train and test Audio Chunk dataset classes.
        """
        meta_data_frame = self.construct_meta_data_file()
        return self.split_into_train_val(train_prop= train_prop)

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

        train_meta_data.to_csv(self.training_meta_dir)
        val_meta_data.to_csv(self.val_meta_dir)

        return self.return_dataset(train_meta_data), self.return_dataset(val_meta_data)

    def retrieve_train_test_split(self):
        """
        Getting train-test split if it already exists in the data folder.
        Otherwise throw exception saying to call the split_into_train_val set.
        """
        train_meta_data = pd.read_csv(self.training_meta_dir)
        val_meta_data = pd.read_csv(self.val_meta_dir)
        return self.return_dataset(train_meta_data), self.return_dataset(val_meta_data)
