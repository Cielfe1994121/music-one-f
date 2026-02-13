from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal  # 畳み込み用


class syn_reverb:
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

    def generate_ir(self, sr, duration=2.5):
        """
        人工的な残響（インパルス応答）を作成するメソッド
        ホワイトノイズに減衰カーブを掛けて「響き」を作ります
        """
        length = int(sr * duration)
        # 1. ホワイトノイズ生成
        ir = np.random.randn(length)

        # 2. 減衰カーブ（指数関数的に音が小さくなる）
        # 最後のほうはゼロになるように
        decay = np.exp(-np.linspace(0, 12, length))

        # 3. ノイズにカーブを適用
        ir = ir * decay
        return ir

    def syn_rev(self, data, sr):
        self.be_data = data.copy()
        self.sr = sr

        if data.ndim == 1:
            data = np.vstack([data, data])
        elif np.mean(np.abs(data[1])) < 0.0001:
            data[1] = data[0].copy()

        length = data.shape[1]

        # ---------------------------------------------------------
        # 1. 残響音（Wet成分）を作る
        # ---------------------------------------------------------
        # 毎回IRを作ると重いので、固定の「綺麗なホール」を作る
        ir = self.generate_ir(sr, duration=3.0)  # 3秒の残響

        # 高速畳み込み（FFTConvolve）
        # 通常のconvolveだと日が暮れるほど遅いので、fftconvolveは必須
        wet_left = signal.fftconvolve(data[0], ir, mode="full")[:length]
        wet_right = signal.fftconvolve(data[1], ir, mode="full")[:length]

        # Wet成分の音量を整える（原音と同じくらいのパワーにする）
        wet_left = wet_left / np.max(np.abs(wet_left))
        wet_right = wet_right / np.max(np.abs(wet_right))

        wet_signal = np.vstack([wet_left, wet_right])

        # ---------------------------------------------------------
        # 2. 1/fゆらぎで「残響の深さ」を変える
        # ---------------------------------------------------------
        self.one_f = ofg.generate_one_f(length)
        raw_mix = self.one_f.ifft_real_result

        # 移動平均（ノイズ対策：Reverbの切り替えもゆっくりが鉄則）
        window_size = 2000
        window = np.ones(window_size) / window_size
        smooth_mix = np.convolve(raw_mix, window, mode="same")

        # 正規化 & スケーリング
        # 0.0(Dry) 〜 0.6(Wet 60%) くらいの間を揺らがせる
        # Reverb成分が100%になるとお風呂すぎて何かわからなくなるので抑えめに
        smooth_mix = (smooth_mix - np.mean(smooth_mix)) * 2.0  # 振幅調整
        mix_ratio = np.clip(smooth_mix + 0.3, 0.0, 0.6)

        # ---------------------------------------------------------
        # 3. ブレンド（Dry + Wet）
        # ---------------------------------------------------------
        # 原音(data) と 残響音(wet_signal) を mix_ratio で混ぜる
        processed_data = data * (1 - mix_ratio) + wet_signal * mix_ratio

        # 最終ノーマライズ（Reverbは音量が足されて膨らむので必須）
        max_val = np.max(np.abs(processed_data))
        if max_val > 0:
            processed_data = processed_data / max_val

        self.af_data = processed_data
        return processed_data

    def vid(self):
        if self.be_data is None or self.af_data is None:
            return

        # 5秒分切り出し
        start_sec = 30
        duration = 5
        s_idx = int(start_sec * self.sr)
        e_idx = int((start_sec + duration) * self.sr)

        # 配列外参照防止
        if e_idx > self.be_data.shape[1]:
            s_idx = 0
            e_idx = int(5 * self.sr)

        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("Reverb Modulation (Dry vs Wet)")

        # 原音
        ax[0].plot(
            self.be_data[0, s_idx:e_idx],
            label="Dry (Original)",
            color="tab:blue",
            alpha=0.8,
        )
        ax[0].set_title("Dry Signal")
        ax[0].legend()

        # 残響あり
        ax[1].plot(
            self.af_data[0, s_idx:e_idx],
            label="Wet Mix (Processed)",
            color="tab:purple",
            alpha=0.8,
        )
        ax[1].set_title("Reverb Added (Notice the 'tail' filling the gaps)")
        ax[1].legend()

        # 比較（残響成分の可視化）
        # 原音が消えたところ（無音部分）に残響が残っているかを見る
        ax[2].plot(
            self.af_data[0, s_idx:e_idx],
            color="tab:purple",
            alpha=0.6,
            label="With Reverb",
        )
        ax[2].plot(
            self.be_data[0, s_idx:e_idx], color="tab:blue", alpha=0.4, label="Original"
        )
        ax[2].set_title("Overlay (Purple tails should extend beyond Blue)")
        ax[2].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    syn = syn_reverb()
    data, sr = syn.get_file_path()

    if data is not None:
        af = syn.syn_rev(data, sr)
        syn.vid()
        # syn.file.play_from_array(af.T, sr)
