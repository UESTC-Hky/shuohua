import numpy as np
import scipy.fftpack
import wave


def read_wav(filepath):
    with wave.open(filepath, 'rb') as w:
        nchannels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        data = w.readframes(nframes)
    samples = np.frombuffer(data, dtype=np.int16)
    samples = samples.astype(np.float64)
    return samples, framerate


def pre_emphasis(signal, alpha=0.97):
    output = np.copy(signal)
    output[1:] = signal[1:] - alpha * signal[:-1]
    return output


def framing(signal, frame_size, frame_shift):
    num_frames = 1 + max(0, (len(signal) - frame_size) // frame_shift)
    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        start = i * frame_shift
        frames[i, :] = signal[start:start + frame_size]
    return frames


def hamming_window(frame_size):
    n = np.arange(frame_size)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (frame_size - 1))


def apply_window(frames):
    window = hamming_window(frames.shape[1])
    return frames * window


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(num_filters, fft_size, sample_rate):
    f_low = 0.0
    f_high = sample_rate / 2.0

    mel_low = hz_to_mel(f_low)
    mel_high = hz_to_mel(f_high)

    mel_points = np.linspace(mel_low, mel_high, num_filters + 2)
    freq_points = mel_to_hz(mel_points)

    bins = np.floor((fft_size + 1) * freq_points / sample_rate).astype(int)

    fbank = np.zeros((num_filters, fft_size // 2 + 1))
    for i in range(1, num_filters + 1):
        for k in range(bins[i - 1], bins[i]):
            fbank[i - 1, k] = (k - bins[i - 1]) / max(bins[i] - bins[i - 1], 1)
        for k in range(bins[i], bins[i + 1]):
            fbank[i - 1, k] = (bins[i + 1] - k) / max(bins[i + 1] - bins[i], 1)

    return fbank


def extract_mfcc(signal, sample_rate, frame_size=256, frame_shift=128,
                 num_filters=26, num_ceps=12, fft_size=512):
    signal_pe = pre_emphasis(signal)
    frames = framing(signal_pe, frame_size, frame_shift)
    frames = apply_window(frames)

    mag_spec = np.abs(np.fft.rfft(frames, n=fft_size))
    pow_spec = mag_spec ** 2 / fft_size

    fbank = mel_filterbank(num_filters, fft_size, sample_rate)
    filter_energies = np.dot(pow_spec, fbank.T)
    filter_energies = np.where(filter_energies == 0, np.finfo(float).eps, filter_energies)
    log_energies = np.log(filter_energies)

    mfcc = scipy.fftpack.dct(log_energies, type=2, axis=1, norm='ortho')[:, :num_ceps]

    return mfcc
