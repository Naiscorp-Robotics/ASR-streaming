import torch
from utils import DecodedResult, logger

def compute_stats_audio(audio: torch.Tensor([]), offset: float, noise_previous: torch.Tensor([]), decoded_result: DecodedResult, sr=8000, use_previous=False):
    """
    Args:
        audio: Torch tensor. (N,)
        noise_previous: Tensor have total segment noise. (N,)
        decoded_result: Result from model.
    Return:
        decoded_result: Update statics information.
        noise_previous: Tensor have total segment noise. (N,)
    """
    segment_start = int((decoded_result.segment_start - offset)*sr)
    segment_end   = int((decoded_result.segment_start + decoded_result.segment_length - offset)*sr)
    speech = torch.Tensor([])
    internal_noise = torch.Tensor([])
    word_alignment = decoded_result.result["hypotheses"][0]["word_alignment"]
    for i, w_a in enumerate(word_alignment):
        word_start = int((w_a["start"] - offset) * sr)
        word_end   = int((w_a["start"] + w_a["length"] - offset) * sr)
        speech = torch.cat((speech, audio[word_start: word_end]))
        if i == 0:
            previous_word_end = word_end
            word_start_ = word_start
        else:
            internal_noise = torch.cat((internal_noise, audio[previous_word_end: word_start]))
            previous_word_end = word_end
        if i == len(word_alignment) - 1:
            word_end_ = word_end

        word_power = ((audio[word_start: word_end]) ** 2).mean() + 1e-9
        vol_word = 10 * torch.log10(word_power)
        logger.debug(f'Volume word: {vol_word}, {w_a}')

    if use_previous:
        noise  = torch.cat((noise_previous, audio[:word_start_], internal_noise, audio[word_end_:]))
    else:
        noise  = torch.cat((audio[segment_start:word_start_], internal_noise, audio[word_end_:segment_end]))

    speech_power = (speech ** 2).mean() + 1e-9
    noise_power  = (noise ** 2).mean() + 1e-9

    SNR        = 10 * torch.log10(speech_power / noise_power)
    vol_noise  = 10 * torch.log10(noise_power)
    vol_speech = 10 * torch.log10(speech_power)

    decoded_result.snr        = round(SNR.item(), 2)
    decoded_result.vol_speech = round(vol_speech.item(), 2)
    decoded_result.vol_noise  = round(vol_noise.item(), 2)

    return decoded_result, noise