import os
from pathlib import Path
import torch
import torchaudio
import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 参数
IMG_SIZE = 460
SR = 32000

# 低频 STFT 参数
Nfft_lf = 32768
Nskip_lf = Nfft_lf // 5
rng_lf = Nskip_lf * (IMG_SIZE - 1)

# 高频 STFT 参数
Nfft_hf = 1024
Nskip_hf = (1024 * 3) // 4
rng_hf = Nskip_hf * IMG_SIZE + Nfft_hf

# Mel 参数
N_MELS = IMG_SIZE  # Mel 频谱通道数 = 460，保持统一尺寸

# 频谱变换器
stft_lf = torchaudio.transforms.Spectrogram(
    n_fft=Nfft_lf, hop_length=Nskip_lf, power=2, return_complex=False).to(device)

stft_hf = torchaudio.transforms.Spectrogram(
    n_fft=Nfft_hf, hop_length=Nskip_hf, power=2, return_complex=False).to(device)

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=Nfft_lf, hop_length=Nskip_lf, n_mels=N_MELS, power=2).to(device)

# 归一化函数
def normSpec(spec):
    ret = torch.log10(spec + 1e-20)
    mean = torch.mean(ret)
    std = torch.std(ret)
    ret = (ret - mean) / (std * 4) + 0.5
    return torch.clamp(ret, 0, 1)

# 读取wav，中心剪切
def readWav_center(p: Path):
    info = torchaudio.info(str(p))
    frames = info.num_frames
    if frames < rng_lf:
        wav = torch.Tensor()
        while wav.shape[-1] < rng_lf:
            wav = torch.cat((wav, torchaudio.load(str(p))[0]), dim=1)
    else:
        wav, _ = torchaudio.load(str(p))
    start = max(0, (wav.shape[-1] - rng_lf) // 2)
    wav = wav[:, start:start + rng_lf]
    return wav.to(device)

# wav转5通道频谱图
def wavToSpecs_center(wav):
    # 低频 STFT
    lf = stft_lf(wav)[0]
    lf0 = normSpec(lf[:IMG_SIZE, :IMG_SIZE])
    lf1 = normSpec(lf[IMG_SIZE:IMG_SIZE * 2, :IMG_SIZE])

    # 高频 STFT
    hf = stft_hf(wav[:, :rng_hf])[0]
    hf = normSpec(hf[12:IMG_SIZE + 12, :IMG_SIZE])

    # Mel 频谱
    mel = mel_transform(wav)[0]
    mel = normSpec(mel[:IMG_SIZE, :IMG_SIZE])

    # CQT 频谱（使用 librosa）
    wav_np = wav.squeeze(0).cpu().numpy()
    cqt = librosa.cqt(wav_np, sr=SR, hop_length=Nskip_lf, n_bins=IMG_SIZE, bins_per_octave=IMG_SIZE // 4)
    cqt = np.abs(cqt)
    cqt = torch.from_numpy(cqt).to(device)
    cqt = normSpec(cqt[:IMG_SIZE, :IMG_SIZE])

    return torch.stack((lf0, lf1, hf, mel, cqt), 0)  # [5, H, W]

# 单文件处理
def process_one_file(f: Path, src_dir: Path, dst_dir: Path):
    try:
        wav = readWav_center(f)
        spec = wavToSpecs_center(wav).cpu().numpy()
        rel_path = f.relative_to(src_dir).with_suffix('.npy')
        save_path = dst_dir / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, spec)
    except Exception as e:
        print(f"Error processing {f}: {e}")

# 批量处理
def preprocess_save_npy(src_dir: Path, dst_dir: Path, max_workers=8):
    src_files = list(src_dir.rglob('*.wav'))
    print(f'Processing {len(src_files)} files...')
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_file, f, src_dir, dst_dir) for f in src_files]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

# -------------------
if __name__ == '__main__':
    src_path = Path('./datasets')      # 原始 wav 路径
    dst_path = Path('./npy_specs_center')  # 保存 npy 路径
    preprocess_save_npy(src_path, dst_path)
