from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal  # 畳み込み用


class syn_reverb:
    def get_file_path(self):
        self.file = gp()
        self.file_path = self.file.gui_get_music()
        # ここでは読み込むだけ
        self.data, self.sr = librosa.load(
            self.file_path, mono=False, sr=None, duration=180
        )
        return self.data, self.sr

    def syn_rev(self, data, sr):
        # 1. 安全装置：モノラル対策
        if data.ndim == 1:
            data = np.vstack([data, data])
        elif np.mean(np.abs(data[1])) < 0.0001:
            print("Mono detected, copying...")
            data[1] = data[0].copy()

        length = data.shape[1]

        # -----------------------------------------------------
        # 2. 人工リバーブ（Impulse Response）作成
        # -----------------------------------------------------
        reverb_time = 2.0  # 残響の長さ
        ir_length = int(sr * reverb_time)

        # ノイズ生成と減衰
        impulse_response = np.random.randn(ir_length)
        decay = np.exp(-np.linspace(0, 5, ir_length))
        impulse_response = impulse_response * decay

        # -----------------------------------------------------
        # 3. 畳み込み（Convolution）でWet音作成
        # -----------------------------------------------------
        print("Convolving reverb... (Please wait)")
        wet_signal = np.zeros_like(data)

        # 左右それぞれ計算
        wet_left = signal.fftconvolve(data[0], impulse_response, mode="full")
        wet_right = signal.fftconvolve(data[1], impulse_response, mode="full")

        # 長さを元データに合わせる
        wet_signal[0] = wet_left[:length]
        wet_signal[1] = wet_right[:length]

        # 正規化（Dryと同じレベルに合わせる）
        wet_max = np.max(np.abs(wet_signal))
        if wet_max > 0:
            wet_signal = wet_signal / wet_max * np.max(np.abs(data))

        # -----------------------------------------------------
        # 4. 1/fゆらぎで混ぜる（Mix）
        # -----------------------------------------------------
        self.one_f = ofg.generate_one_f(length)

        # 混ぜ具合（0.0〜0.6くらい）
        mix_ratio = (self.one_f.ifft_real_result - 1) * 0.3
        mix_ratio = np.clip(mix_ratio + 0.2, 0.0, 0.6)

        # ブレンド
        processed_data = data * (1 - mix_ratio) + wet_signal * mix_ratio

        return processed_data

    def vid(self, be, af):
        # 1秒分だけ表示
        self.limit = int(1 * self.sr)
        fig, ax = plt.subplots(3, 1, sharex=True)

        ax[0].plot(be[0, : self.limit], label="Before (Dry)", color="tab:blue")
        ax[1].plot(af[0, : self.limit], label="After (Reverb Mix)", color="tab:purple")

        ax[2].plot(be[0, : self.limit], label="Before", color="tab:blue", alpha=0.6)
        ax[2].plot(af[0, : self.limit], label="After", color="tab:purple", alpha=0.6)

        for a in ax:
            # ズーム自動
            a.legend(loc="upper right")
            a.grid(True, linestyle="--", alpha=0.5)

        plt.show()


if __name__ == "__main__":
    syn = syn_reverb()

    # 1. 読み込み
    data, sr = syn.get_file_path()

    # 比較用コピー
    be = data.copy()

    # 2. 加工（少し時間がかかります）
    af = syn.syn_rev(data, sr)

    # 3. 表示
    syn.vid(be, af)

    # 再生
    # syn.file.play_from_array(af.T, sr)
