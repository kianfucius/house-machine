from auraloss.freq import STFTLoss, SumAndDifference
import torch
import torch.nn.functional as F
from torch import nn
from typing import List

class CustomFrequencyLoss(nn.Module):
    """
    Class for custom pytorch loss with Custom frequency loss penalty term.
    """

    def __init__(self, frequency_weight = 0.001,        
                fft_sizes: List[int] = [2048, 4096, 1024],
                hop_sizes: List[int] = [120, 240, 50],
                win_lengths: List[int] = [600, 1200, 240], 
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frequency_loss = CustomSumAndDifferenceSTFTLoss(fft_sizes=fft_sizes,
                                                       hop_sizes = hop_sizes,
                                                       win_lengths=win_lengths,
                                                       reduction = 'none')
        self.weighting =frequency_weight
        

    def forward(self, 
                v_pred:torch.Tensor, 
                v_true:torch.Tensor, 
                x_pred:torch.Tensor, 
                x_true:torch.Tensor,
                alphas:torch.Tensor,
                sigmas:torch.Tensor,
                tuple_loss= False):
        assert sigmas.shape == alphas.shape, 'Sigmas and Alphas have different shape'
        mean_loss = F.mse_loss(v_pred,v_true) 
        frequency_loss = self.frequency_loss(x_true, x_pred,alphas, sigmas)
        if tuple_loss:
            return mean_loss,frequency_loss
        else:
            return mean_loss + frequency_loss*self.weighting
    


class CustomMultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module (with diffusion weights)
    Taken from auraloss library and adapted to
    See [Yamamoto et al., 2019](https://arxiv.org/abs/1910.11480)

    Args:
        fft_sizes (list): List of FFT sizes.
        hop_sizes (list): List of hop sizes.
        win_lengths (list): List of window lengths.
        window (str, optional): Window to apply before FFT, options include:
            'hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag (float, optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of mel frequency bins. Required when scale = 'mel'. Default: None.
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
    """

    def __init__(
        self,
        fft_sizes: List[int] = [1024, 2048, 512],
        hop_sizes: List[int] = [120, 240, 50],
        win_lengths: List[int] = [600, 1200, 240],
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float = None,
        scale: str = None,
        n_bins: int = None,
        perceptual_weighting: bool = False,
        scale_invariance: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [
                STFTLoss(
                    fs,
                    ss,
                    wl,
                    window,
                    w_sc,
                    w_log_mag,
                    w_lin_mag,
                    w_phs,
                    sample_rate,
                    scale,
                    n_bins,
                    perceptual_weighting,
                    scale_invariance,
                    **kwargs,
                )
            ]

    def forward(self, x, y, alphas, sigmas):
        mrstft_loss = 0.0

        for f in self.stft_losses:
            mrstft_loss += torch.mean(f(x, y)*(1+ (alphas/sigmas)**2))

        mrstft_loss /= len(self.stft_losses)

        return mrstft_loss
        

class CustomSumAndDifferenceSTFTLoss(torch.nn.Module):
    """Sum and difference sttereo STFT loss module.

    See [Steinmetz et al., 2020](https://arxiv.org/abs/2010.10291)

    Args:
        fft_sizes (List[int]): List of FFT sizes.
        hop_sizes (List[int]): List of hop sizes.
        win_lengths (List[int]): List of window lengths.
        window (str, optional): Window function type.
        w_sum (float, optional): Weight of the sum loss component. Default: 1.0
        w_diff (float, optional): Weight of the difference loss component. Default: 1.0
        perceptual_weighting (bool, optional): Apply perceptual A-weighting (Sample rate must be supplied). Default: False
        mel_stft (bool, optional): Use Multi-resoltuion mel spectrograms. Default: False
        n_mel_bins (int, optional): Number of mel bins to use when mel_stft = True. Default: 128
        sample_rate (float, optional): Audio sample rate. Default: None
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'
    """

    def __init__(
        self,
        fft_sizes: List[int],
        hop_sizes: List[int],
        win_lengths: List[int],
        window: str = "hann_window",
        w_sum: float = 1.0,
        w_diff: float = 1.0,
        output: str = "loss",
        **kwargs,
    ):
        super().__init__()
        self.sd = SumAndDifference()
        self.w_sum = w_sum
        self.w_diff = w_diff
        self.output = output
        self.mrstft = CustomMultiResolutionSTFTLoss(
            fft_sizes,
            hop_sizes,
            win_lengths,
            window,
            **kwargs,
        )

    def forward(self, input: torch.Tensor, 
                target: torch.Tensor,
                alphas:torch.Tensor, 
                sigmas:torch.Tensor):
        
        """This loss function assumes batched input of stereo audio in the time domain.

        Args:
            input (torch.Tensor): Input tensor with shape (batch size, 2, seq_len).
            target (torch.Tensor): Target tensor with shape (batch size, 2, seq_len).

        Returns:
            loss (torch.Tensor): Aggreate loss term. Only returned if output='loss'.
            loss (torch.Tensor), sum_loss (torch.Tensor), diff_loss (torch.Tensor):
                Aggregate and intermediate loss terms. Only returned if output='full'.
        """
        assert input.shape == target.shape  # must have same shape
        bs, chs, seq_len = input.size()

        # compute sum and difference signals for both
        input_sum, input_diff = self.sd(input)
        target_sum, target_diff = self.sd(target)

        # compute error in STFT domain
        sum_loss = self.mrstft(input_sum, target_sum, alphas, sigmas)
        diff_loss = self.mrstft(input_diff, target_diff, alphas, sigmas)
        return ((self.w_sum * sum_loss) + (self.w_diff * diff_loss)) / 2