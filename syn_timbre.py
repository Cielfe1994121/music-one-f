from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal  # フィルター用


class syn_timbre:
    def __init__(self):
        self.sr = 44100
        self.be_data = None
        self.af_data = None

    def get_file_path(self):
        self.file = gp()
        self.file_path = self.file.gui_get_music()
        if not self.file_path:
            return None, None

        self.data, self.sr = librosa.load(
            self.file_path, mono=False, sr=None, duration=180
        )
        return self.data, self.sr

    def syn_tim(self, data, sr):
        # グラフ比較用に保存
        self.be_data = data.copy()
        self.sr = sr  # srを確実に保存

        # 1. 安全装置
        if data.ndim == 1:
            data = np.vstack([data, data])
        elif np.mean(np.abs(data[1])) < 0.0001:
            data[1] = data[0].copy()

        length = data.shape[1]

        # 2. フィルター作成
        nyquist = sr / 2
        cutoff = 1000 / nyquist
        b, a = signal.butter(4, cutoff, btype="low")

        muffled_data = np.zeros_like(data)
        # filtfiltで位相ズレなし
        muffled_data[0] = signal.filtfilt(b, a, data[0])
        muffled_data[1] = signal.filtfilt(b, a, data[1])

        # 3. 1/fゆらぎ係数
        self.one_f = ofg.generate_one_f(length)
        raw_ratio = self.one_f.ifft_real_result

        # 移動平均（ノイズ対策）
        window_size = 2000
        window = np.ones(window_size) / window_size
        smooth_ratio = np.convolve(raw_ratio, window, mode="same")

        # 正規化
        smooth_ratio = (smooth_ratio - np.mean(smooth_ratio)) * 8.0
        mix_ratio = np.clip(smooth_ratio + 0.3, 0.0, 1.0)

        # 4. ブレンド
        processed_data = data * (1 - mix_ratio) + muffled_data * mix_ratio

        self.af_data = processed_data

        return processed_data

    def vid(self):
        if self.be_data is None or self.af_data is None:
            return

        # ---------------------------------------------------------
        # 【修正】重すぎて固まるので、5秒分だけ切り出して分析する
        # ---------------------------------------------------------
        start_sec = 30  # 曲の30秒目から（イントロ終わり付近）
        duration = 5  # 5秒間だけ見る

        s_idx = int(start_sec * self.sr)
        e_idx = int((start_sec + duration) * self.sr)

        # データが短い場合の安全策
        if e_idx > self.be_data.shape[1]:
            s_idx = 0
            e_idx = int(5 * self.sr)

        chunk_be = self.be_data[0, s_idx:e_idx]
        chunk_af = self.af_data[0, s_idx:e_idx]

        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Timbre Fluctuation (Spectrogram) - {duration}s slice")

        # スペクトログラム計算（切り出したデータで行う）
        f_be, t_be, Sxx_be = signal.spectrogram(chunk_be, self.sr, nperseg=1024)
        f_af, t_af, Sxx_af = signal.spectrogram(chunk_af, self.sr, nperseg=1024)

        # 上段：Before
        ax[0].pcolormesh(
            t_be, f_be, 10 * np.log10(Sxx_be + 1e-10), shading="gouraud", cmap="inferno"
        )
        ax[0].set_ylabel("Frequency [Hz]")
        ax[0].set_title("Original")
        ax[0].set_ylim(0, 5000)

        # 下段：After
        im = ax[1].pcolormesh(
            t_af, f_af, 10 * np.log10(Sxx_af + 1e-10), shading="gouraud", cmap="inferno"
        )
        ax[1].set_ylabel("Frequency [Hz]")
        ax[1].set_xlabel("Time [sec]")
        ax[1].set_title('Processed (Look for "Cloudy" changes)')
        ax[1].set_ylim(0, 5000)

        fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.05, pad=0.05)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    syn = syn_timbre()
    data, sr = syn.get_file_path()

    if data is not None:
        af = syn.syn_tim(data, sr)
        syn.vid()
