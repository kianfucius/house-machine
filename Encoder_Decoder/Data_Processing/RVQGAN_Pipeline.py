import dac
from audiotools import AudioSignal
import torch


class RVQGANEncoder:
    """
    Wrapper class for RVQGAN functionality.
    """

    def __init__(self, tensor_len) -> None:
        model_path = dac.utils.download(model_type="44khz")
        self.model_sample_rate = 44100
        self.model = dac.DAC.load(model_path)
        self.model.to('cuda')
        self.audio_seconds=  self.get_seconds_for_tensor_len(tensor_len)
        self.encoded_len = self.length_after_encoding(tensor_len)
        self.latent_dim = 72

    def get_seconds_for_tensor_len(self, desired_tensor_len):
        """
        Converting tensor length argument to seconds to know how much to read from audio file.
        """
        x= desired_tensor_len/72
        x = x/0.0019805361871987857
        x= x/(44100/48000)
        x = x/48000
        return x

    def length_after_encoding(self,  tensor_len):
        """
        Calculating length of time dimension (last dimension) after tensor encoding.
        """
        return tensor_len*0.0019805361871987857


    @torch.inference_mode()
    def get_latents(self, x:AudioSignal) -> torch.tensor:
        """
        Get latent representation of audio_tensor.
        """
        x.to(self.model.device)
        if x.sample_rate != self.model_sample_rate:
            x = x.resample(self.model_sample_rate)
        x = x.to_mono()
        x = self.model.preprocess(x.audio_data,x.sample_rate)
        return self.model.encode(x)[2]
    
    @torch.inference_mode()
    def decode_and_save_audio_from_latents(self, x:torch.tensor, filename:str)->None:
        """
        Saves Audio as a wave file from latent tensor.
    
        """
        x= self.model.quantizer.from_latents(x)[0]
        x = AudioSignal(self.model.decode(x),
                                    sample_rate = 44100)
        loudness_fn = x.loudness

        x.normalize(-16)
        #recons = recons[..., : obj.original_length]
        loudness_fn()
        x.audio_data = x.audio_data.detach().to('cpu')
        x.write(filename)