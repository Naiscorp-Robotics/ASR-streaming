import os
import torch

print(os.environ["TORCH_THREAD"])
print(os.environ.get("LANGUAGE", "vi"))



# TEST VAD SILERO
sample_rate = 16000
torch.set_num_threads(1)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model="silero_vad")
(get_speech_timestamps, _, read_audio, _, _) = utils

wav = read_audio("test.wav", sampling_rate=sample_rate)
speech_timestamps = get_speech_timestamps(wav, model, threshold=0.5, sampling_rate=sample_rate, return_seconds=True)
print(speech_timestamps)

a = torch.linspace(0,23, steps=24)
a = a.reshape(2,3,4)
print(a)
lens = [x.size(0) for x in a]
print(torch.tensor(lens))

s = [torch.linspace(0,23, steps=24).reshape(6,1,4) for _ in range(20)]
s = torch.cat(s, dim=1)
print(s.shape)

y = s[:, 1,:]
print(y.shape)