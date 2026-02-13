from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa
import matplotlib.pyplot as plt
import numpy as np


class syn_pan:
    def __init__(self):
        self.sr = 44100
        # グラフ表示用にデータを保持する変数を初期化
        self.data = None

    def get_file_path(self):
        self.file = gp()
        self.file_path = self.file.gui_get_music()
        if not self.file_path:
            return None, None

        self.data, self.sr = librosa.load(
            self.file_path, mono=False, sr=None, duration=180
        )
        return self.data, self.sr

    def syn_pan(self, data, sr):
        # 1. モノラル対策
        if data.ndim == 1:
            data = np.vstack([data, data])
        elif np.mean(np.abs(data[1])) < 0.0001:
            # print("Mono source detected! Copying Left to Right...")
            data[1] = data[0].copy()

        length = data.shape[1]

        # 1/fゆらぎ生成
        self.one_f = ofg.generate_one_f(length)
        raw_pan = self.one_f.ifft_real_result

        # ---------------------------------------------------------
        # 【修正1】ゆらぎを滑らかにする（ざざざノイズ対策）
        # ---------------------------------------------------------
        window_size = 50
        window = np.ones(window_size) / window_size
        pan_signal = np.convolve(raw_pan, window, mode="same")

        # ---------------------------------------------------------
        # 【修正2】数値を 0.0(左) 〜 1.0(右) に強制的に収める
        # ---------------------------------------------------------
        # これをやらないと、数値が偏って「ずっと右」になります

        # まず中心(平均)を0に持ってくる
        pan_signal = pan_signal - np.mean(pan_signal)

        # 最大値で割って -1.0 〜 1.0 にする
        max_val = np.max(np.abs(pan_signal))
        if max_val > 0:
            pan_signal = pan_signal / max_val

        # 0.0 〜 1.0 に変換（0.5がセンター）
        # ※ 0.8を掛けているのは、完全に0や1になると片耳が聞こえなくなるのを防ぐため
        pan_curve = (pan_signal * 0.8 + 1.0) / 2.0

        # ---------------------------------------------------------
        # 【修正3】左右に割り振る（Constant Power Panning）
        # ---------------------------------------------------------
        # 単純な足し算ではなく sqrt を使うと、真ん中でも音が痩せません
        left_gain = np.sqrt(1.0 - pan_curve)
        right_gain = np.sqrt(pan_curve)

        # 元データを書き換える
        pan_data = data.copy()
        pan_data[0] = data[0] * left_gain
        pan_data[1] = data[1] * right_gain

        # 【重要】グラフ表示用に、加工後のデータをクラス変数に保存しておく
        self.data = pan_data

        return pan_data

    def get_lfri(self):
        # 加工後のデータがあればそれを返す
        if self.data is not None:
            return self.data[0], self.data[1]
        else:
            return np.array([]), np.array([])

    def vid(self, lf, ri):
        # データが空なら何もしない
        if len(lf) == 0:
            return

        self.limit = int(1 * self.sr)

        # グラフの準備
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        fig.suptitle("1/f Panning Visualization (Left vs Right)")

        # 上段：左チャンネル
        ax[0].plot(lf[: self.limit], label="Left Channel", color="tab:blue")
        ax[0].set_ylabel("Amplitude")
        ax[0].legend(loc="upper right")
        ax[0].grid(True, linestyle="--", alpha=0.5)

        # 中段：右チャンネル
        ax[1].plot(ri[: self.limit], label="Right Channel", color="tab:orange")
        ax[1].set_ylabel("Amplitude")
        ax[1].legend(loc="upper right")
        ax[1].grid(True, linestyle="--", alpha=0.5)

        # 下段：重ね合わせ（動きの違いを見る）
        ax[2].plot(lf[: self.limit], label="Left", color="tab:blue", alpha=0.8)
        ax[2].plot(ri[: self.limit], label="Right", color="tab:orange", alpha=0.6)
        ax[2].set_ylabel("Overlay")
        ax[2].set_xlabel("Samples")
        ax[2].legend(loc="upper right")
        ax[2].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    syn = syn_pan()

    # ファイル読み込み
    data, sr = syn.get_file_path()

    if data is not None:
        # 加工実行
        processed_data = syn.syn_pan(data, sr)

        # 左右チャンネルを取得して可視化
        syn_lf, syn_ri = syn.get_lfri()
        syn.vid(syn_lf, syn_ri)

        # 再生テスト（必要なら）
        # syn.file.play_from_array(processed_data.T, sr)
