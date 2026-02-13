from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa
import matplotlib.pyplot as plt
import numpy as np


class syn_volume:
    def __init__(self):
        # 初期値（エラー回避用）
        self.sr = 44100

    def get_file_path(self):
        self.file = gp()
        self.file_path = self.file.gui_get_music()
        if not self.file_path:
            return None, None

        # 他のクラスと統一してステレオ(mono=False)で読み込む
        self.data, self.sr = librosa.load(
            self.file_path, mono=False, sr=None, duration=180
        )
        return self.data, self.sr

    def syn_vol(self, data, sr):
        # グラフ描画用（vidメソッド）にsrを保存しておく
        self.sr = sr

        # 1. 安全装置：モノラル対策
        if data.ndim == 1:
            data = np.vstack([data, data])
        elif np.mean(np.abs(data[1])) < 0.0001:
            # print("Mono source detected! Copying Left to Right...")
            data[1] = data[0].copy()

        length = data.shape[1]

        # 1/fゆらぎ生成
        self.one_f = ofg.generate_one_f(length)
        raw_multiplier = self.one_f.ifft_real_result

        # ---------------------------------------------------------
        # 【修正ポイント】 "ざざざ" ノイズ対策：移動平均でカドを取る
        # ---------------------------------------------------------
        # window_sizeを大きくするほど滑らかになります（推奨: 50〜100）
        window_size = 50
        window = np.ones(window_size) / window_size

        # 畳み込み積分でスムージング（mode='same'でサイズを変えない）
        multiplier = np.convolve(raw_multiplier, window, mode="same")

        # 必要に応じてオフセット調整（例：極端に音が小さくなるのを防ぐ）
        # multiplier = (multiplier - np.mean(multiplier)) + 1.0

        # 2. 音量揺らぎの適用
        # shapeが違う(2行 vs 1行)ので、numpyが自動で各行に掛けてくれます
        vol_data = data * multiplier

        # 3. 最終安全装置：クリッピング防止（ノーマライズ）
        # 計算結果が1.0を超えて音割れするのを防ぎます
        max_val = np.max(np.abs(vol_data))
        if max_val > 0:
            vol_data = vol_data / max_val

        return vol_data

    def vid(self, be, af):
        # 1秒分だけ表示
        if hasattr(self, "limit"):
            limit = self.limit
        else:
            limit = int(1 * self.sr)

        fig, ax = plt.subplots(3, 1, sharex=True)

        # ステレオ対応：左チャンネル[0]だけをプロット
        ax[0].plot(be[0, :limit], label="Before", color="tab:blue")
        ax[0].set_title("Original")

        ax[1].plot(af[0, :limit], label="After (Smoothed 1/f)", color="tab:orange")
        ax[1].set_title("Processed")

        ax[2].plot(be[0, :limit], label="Before", color="tab:blue", alpha=1)
        ax[2].plot(af[0, :limit], label="After", color="tab:orange", alpha=0.5)
        ax[2].set_title("Overlay")

        for a in ax:
            a.legend(loc="upper right")
            a.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    syn = syn_volume()

    # 1. 読み込み
    data, sr = syn.get_file_path()

    if data is not None:
        # 比較用にコピーをとっておく
        be_data = data.copy()

        # 2. 加工
        af_data = syn.syn_vol(data, sr)

        # 3. 表示
        syn.vid(be_data, af_data)

        # 再生（必要ならコメントアウト解除）
        # syn.file.play_from_array(af_data.T, sr)
