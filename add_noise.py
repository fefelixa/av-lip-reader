from pathlib import Path

import numpy as np
from scipy.io import wavfile


def load_wav_mono(path: Path):
    """Read wav file and convert to mono + float32 in [-1, 1]"""
    sr, data = wavfile.read(str(path))

    # Convert to float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    # Stereo -> Mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    return sr, data


def match_length(noise: np.ndarray, target_len: int, rng: np.random.Generator):
    """Adjust noise length to match speech: repeat if too short, randomly crop if too long"""
    if len(noise) == target_len:
        return noise

    if len(noise) < target_len:
        reps = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, reps)
        return noise[:target_len]
    else:
        start = rng.integers(0, len(noise) - target_len + 1)
        return noise[start:start + target_len]


def mix_with_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, rng=None):
    """
    Mix with specified SNR (dB):
        y[n] = x[n] + alpha * d[n]
    where alpha is computed from global energy.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Align length
    noise = match_length(noise, len(clean), rng)

    # Compute energies for SNR
    Px = np.mean(clean ** 2)
    Pn = np.mean(noise ** 2)
    if Pn == 0:
        raise ValueError("Noise signal has zero power.")

    snr_linear = 10.0 ** (snr_db / 10.0)  # SNR = Px / (alpha^2 * Pn)
    alpha = np.sqrt(Px / (Pn * snr_linear))

    noisy = clean + alpha * noise
    noisy = np.clip(noisy, -1.0, 1.0)  # Prevent clipping overflow

    return noisy


def save_wav(path: Path, data: np.ndarray, sr: int):
    """Save float32[-1,1] to int16 wav"""
    data_int16 = np.int16(np.clip(data, -1.0, 1.0) * 32767)
    wavfile.write(str(path), sr, data_int16)


def add_noise_to_one_file(clean_wav: Path, noise_wav: Path, snr_db: float, out_wav: Path):
    """Add noise to a single wav file and write to a new file"""
    sr_clean, clean = load_wav_mono(clean_wav)
    sr_noise, noise = load_wav_mono(noise_wav)

    if sr_clean != sr_noise:
        raise ValueError(f"Sample rate mismatch: clean={sr_clean}, noise={sr_noise}")

    noisy = mix_with_snr(clean, noise, snr_db)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    save_wav(out_wav, noisy, sr_clean)
    print(f"[INFO] saved noisy wav (SNR={snr_db} dB) -> {out_wav}")


def batch_add_noise(clean_dir: Path, noise_wav: Path, snr_list, out_dir: Path):
    """
    Batch add noise to wav files in a directory and group outputs into subfolders by SNR:
    clean_dir : directory of clean wav files
    noise_wav : single noise wav file
    snr_list  : e.g., [20, 10, 0]
    out_dir   : root output directory (e.g. "wav_noisy")
    """
    clean_dir = Path(clean_dir)
    noise_wav = Path(noise_wav)
    out_dir = Path(out_dir)

    # Read noise file first
    _, noise = load_wav_mono(noise_wav)
    rng = np.random.default_rng()

    wav_paths = sorted(clean_dir.glob("*.wav"))
    print(f"[INFO] found {len(wav_paths)} clean wav files in {clean_dir}")

    for wav_path in wav_paths:
        sr, clean = load_wav_mono(wav_path)

        for snr_db in snr_list:
            noisy = mix_with_snr(clean, noise, snr_db, rng=rng)

            snr_dir = out_dir / f"snr{int(snr_db)}"
            snr_dir.mkdir(parents=True, exist_ok=True)

            out_path = snr_dir / wav_path.name

            save_wav(out_path, noisy, sr)
            print(f"[INFO] {wav_path.name} -> {out_path} (SNR={snr_db} dB)")


if __name__ == "__main__":

    # Modify paths according to your project:
    clean_dir = Path("wav")              # Your clean wav directory
    noise_wav = Path("noise/noise1.wav") # Your noise wav file
    out_dir = Path("wav_noisy")          # Output directory
    snr_list = [20, 10, 0]               # Desired SNR list

    batch_add_noise(clean_dir, noise_wav, snr_list, out_dir)
