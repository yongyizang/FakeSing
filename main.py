from dtw import dtw
import numpy as np
import soundfile as sf
import librosa
import time
from tqdm import tqdm

max_left_latency_s = 0.1  # seconds
max_right_latency_s = 0.1  # seconds
actual_sample_rate = 48000 # Hz
processing_sample_rate = 16000 # Hz
window_size_s = 0.025  # seconds
hop_size_s = 0.01  # seconds

window_size = int(window_size_s * processing_sample_rate)
hop_size = int(hop_size_s * processing_sample_rate)

live_audio = sf.read("live_audio.wav")[0]
reference_audio = sf.read("reference_audio.wav")[0]

audio_length = len(live_audio) / actual_sample_rate

start_time = time.time()

live_audio_downsampled_nonpad = librosa.resample(
    live_audio, orig_sr=actual_sample_rate, target_sr=processing_sample_rate
)
reference_audio_downsampled = librosa.resample(
    reference_audio, orig_sr=actual_sample_rate, target_sr=processing_sample_rate
)

live_audio_downsampled = np.pad(
    live_audio_downsampled_nonpad,
    (int(max_left_latency_s * processing_sample_rate), int(max_right_latency_s * processing_sample_rate)),
    mode="constant",
)
reference_audio_downsampled = np.pad(
    reference_audio_downsampled,
    (int(max_left_latency_s * processing_sample_rate), int(max_right_latency_s * processing_sample_rate)),
    mode="constant",
)

if len(live_audio_downsampled) > len(reference_audio_downsampled):
    reference_audio_downsampled = np.pad(
        reference_audio_downsampled,
        (0, len(live_audio_downsampled) - len(reference_audio_downsampled)),
        mode="constant",
    )
elif len(live_audio_downsampled) < len(reference_audio_downsampled):
    live_audio_downsampled = np.pad(
        live_audio_downsampled,
        (0, len(reference_audio_downsampled) - len(live_audio_downsampled)),
        mode="constant",
    )

max_left_latency = int(max_left_latency_s * processing_sample_rate // window_size)
max_right_latency = int(max_right_latency_s * processing_sample_rate // window_size)

cosine_similarity = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def align_mels(mel_1, mel_2):
    mel_1 = mel_1.T
    mel_2 = mel_2.T
    path = dtw(mel_1, mel_2, dist=cosine_similarity)[3]
    return np.array(path)

live_spectrogram = librosa.stft(
    live_audio_downsampled, n_fft=window_size, hop_length=hop_size
)

live_mel = librosa.feature.melspectrogram(
    S=live_spectrogram, sr=processing_sample_rate, n_mels=40
)

reference_spectrogram = librosa.stft(
    reference_audio_downsampled, n_fft=window_size, hop_length=hop_size
)

reference_mel = librosa.feature.melspectrogram(
    S=reference_spectrogram, sr=processing_sample_rate, n_mels=40
)

reference_fundamental = np.argmax(librosa.power_to_db(reference_mel), axis=0)

reference_fundamental = reference_fundamental.T

correction_values = []
original_values = []

for i in tqdm(range(live_mel.shape[1] - max_left_latency - max_right_latency), desc="Aligning mels"):
    live_mel_slice = live_mel[:, i : i + max_left_latency + max_right_latency]
    reference_mel_slice = reference_mel[:, i : i + max_left_latency + max_right_latency]
    path = align_mels(live_mel_slice, reference_mel_slice)
    target_index = path[0][max_left_latency]
    
    # calculate live_mel's fundamental frequency at this frame
    live_fundamental = np.argmax(librosa.power_to_db(live_mel_slice), axis=0)
    live_fundamental = live_fundamental.T
    live_fundamental = live_fundamental[max_left_latency]
    original_values.append(live_fundamental)
    
    # get reference mel's fundamental frequency at this frame
    reference_fundamental_slice = reference_fundamental[target_index + i]

    correction_values.append(reference_fundamental_slice)

end_time = time.time() - start_time
rtf = end_time / audio_length
print(f"RTF: {rtf}")
    
correction_values = np.array(correction_values)
original_values = np.array(original_values)
frequencies = librosa.mel_frequencies(n_mels=40, fmin=0, fmax=processing_sample_rate / 2)

correction_values = frequencies[correction_values]
original_values = frequencies[original_values]

time_vector = np.arange(0, len(correction_values) * hop_size_s, hop_size_s)

# plot the spectrograms, and overlay the original and corrected fundamental frequencies
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_size_inches(30, 10)
fig.set_dpi(100)
fig.suptitle("Live Vocal Mel Spectrogram - Original")
img = librosa.display.specshow(
    librosa.power_to_db(live_mel, ref=np.max), y_axis="mel", x_axis="time", ax=ax, sr=processing_sample_rate, hop_length=hop_size, fmin=0, fmax=processing_sample_rate / 2, win_length=window_size
)
fig.colorbar(img, ax=ax, format="%+2.0f dB")
ax.plot(
    time_vector,
    original_values,
    color="white",
    label="Original",
)
ax.legend()
plt.savefig("live_mel_original.png")

plt.close()

fig, ax = plt.subplots()
fig.set_size_inches(30, 10)
fig.set_dpi(100)
fig.suptitle("Live Vocal Mel Spectrogram - Corrected")
img = librosa.display.specshow(
    librosa.power_to_db(live_mel, ref=np.max), y_axis="mel", x_axis="time", ax=ax, sr=processing_sample_rate, hop_length=hop_size, fmin=0, fmax=processing_sample_rate / 2, win_length=window_size
)
fig.colorbar(img, ax=ax, format="%+2.0f dB")
ax.plot(
    time_vector,
    correction_values,
    color="white",
    label="Corrected",
)
ax.legend()
plt.savefig("live_mel_corrected.png")

fig, ax = plt.subplots()
fig.set_size_inches(30, 10)
fig.set_dpi(100)
fig.suptitle("Reference Vocal Mel Spectrogram")
img = librosa.display.specshow(
    librosa.power_to_db(reference_mel, ref=np.max), y_axis="mel", x_axis="time", ax=ax, sr=processing_sample_rate, hop_length=hop_size, fmin=0, fmax=processing_sample_rate / 2, win_length=window_size
)
fig.colorbar(img, ax=ax, format="%+2.0f dB")
ax.plot(
    time_vector,
    frequencies[reference_fundamental][: len(correction_values)],
    color="white",
    label="Reference",
)
plt.savefig("reference_mel.png")