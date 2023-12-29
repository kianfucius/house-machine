import dac
from audiotools import AudioSignal
import torch


class RVQGANEncoder:
    """
    Wrapper class for RVQGAN functionality.
    """

    def __init__(self) -> None:
        model_path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(model_path)
        self.model.to('cuda')

    def get_latents(self, x:AudioSignal) -> torch.tensor:
        """
        Get latent representation of audio_tensor.
        """
        x.to(self.model.device)
        x = x.resample(44100)
        x = x.to_mono()
        x = self.model.preprocess(x.audio_data,x.sample_rate)
        return self.model.encode(x)[2]
    
    @torch.no_grad()
    def save_audio_from_latents(self, x:torch.tensor, filename:str)->None:
        """
        Saves Audio as a wave file from latent chunk.
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