from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa
import matplotlib.pyplot as plt
import numpy as np


class syn_pitch:
    def __init__(self):
        self.sr = 44100
        # グラフ用データ保存
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

    def syn_pit(self, data, sr):
        # グラフ比較用に加工前のコピーを取っておく
        self.be_data = data.copy()

        # 1. 安全装置
        if data.ndim == 1:
            data = np.vstack([data, data])
        elif np.mean(np.abs(data[1])) < 0.0001:
            data[1] = data[0].copy()

        length = data.shape[1]

        # 1/fゆらぎ生成
        self.one_f = ofg.generate_one_f(length)
        raw_fluctuation = self.one_f.ifft_real_result

        # ---------------------------------------------------------
        # 【修正1】ゆらぎを「ゆったり」にする（移動平均）
        # ---------------------------------------------------------
        # Pitchは変化が急だと「ビブラート」になり、酔います。
        # 変化を遅くして「ワウ（うねり）」にします。
        window_size = 2000  # かなり大きくして、変化をゆっくりにする
        window = np.ones(window_size) / window_size
        smooth_fluctuation = np.convolve(raw_fluctuation, window, mode="same")

        # ---------------------------------------------------------
        # 【修正2】係数の微調整（ここがエモさの肝）
        # ---------------------------------------------------------
        # 1.0 はやりすぎ。0.002 〜 0.005 くらいが Lo-Fi の黄金比です。
        depth = 0.003

        # 中心を0にしてからスケーリング
        smooth_fluctuation = smooth_fluctuation - np.mean(smooth_fluctuation)

        # 時間軸の「歪みマップ」を作る
        # 1.0 = 通常速度, 1.003 = 少し速い, 0.997 = 少し遅い
        speed_map = 1.0 + (smooth_fluctuation * depth)

        # 累積和をとって「再生位置（インデックス）」に変換
        dirty_time_index = np.cumsum(speed_map)

        # 尺合わせ（曲の長さに強制的に戻す）
        # これをしないと曲の長さが変わってバケツリレーが壊れます
        dirty_time_index = dirty_time_index / dirty_time_index[-1] * (length - 1)

        # ---------------------------------------------------------
        # リサンプリング（補間）
        # ---------------------------------------------------------
        original_index = np.arange(length)

        # 左右で「同じゆらぎ」を適用（位相ズレを防ぐため）
        # ※左右で違うゆらぎにすると、位相がおかしくなって気持ち悪くなります
        data[0] = np.interp(dirty_time_index, original_index, data[0])
        data[1] = np.interp(dirty_time_index, original_index, data[1])

        # 加工後データを保存（グラフ用）
        self.af_data = data

        return data

    def vid(self):
        # データが無いなら何もしない
        if self.be_data is None or self.af_data is None:
            return

        # Pitch変化は波形だと分かりにくいので、「ズレ」を可視化します
        # 冒頭の0.1秒だけを拡大して、波形がどう「伸び縮み」したか見ます
        limit = int(0.1 * self.sr)
        start = int(1.0 * self.sr)  # 1秒地点から表示

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        fig.suptitle("Pitch/Time Fluctuation Analysis (Zoom 0.1s)")

        # 上段：波形の重ね合わせ
        # 青とオレンジが「左右にズレている」のが見えたら成功（時間が歪んでいる証拠）
        ax[0].plot(
            self.be_data[0, start : start + limit],
            label="Original",
            color="tab:blue",
            alpha=0.8,
        )
        ax[0].plot(
            self.af_data[0, start : start + limit],
            label="Pitch Modulated",
            color="tab:orange",
            alpha=0.6,
        )
        ax[0].set_title("Waveform Shift (Time Warping)")
        ax[0].legend(loc="upper right")

        # 下段：差分（どれくらいズレたか）
        diff = (
            self.af_data[0, start : start + limit]
            - self.be_data[0, start : start + limit]
        )
        ax[1].plot(diff, label="Difference (Error)", color="green", alpha=0.5)
        ax[1].set_title("Difference Magnitude")
        ax[1].legend(loc="upper right")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    syn = syn_pitch()
    data, sr = syn.get_file_path()

    if data is not None:
        data = syn.syn_pit(data, sr)
        syn.vid()
        # syn.file.play_from_array(data.T, sr)
