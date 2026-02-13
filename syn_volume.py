from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa
import matplotlib.pyplot as plt
import numpy as np


class syn_volume:
    def get_file_path(self):
        self.file = gp()
        self.file_path = self.file.gui_get_music()
        # 他のクラスと統一してステレオ(mono=False)で読み込む
        self.data, self.sr = librosa.load(
            self.file_path, mono=False, sr=None, duration=180
        )
        return self.data, self.sr

    def syn_vol(self, data, sr):
        # 1. 安全装置：モノラル対策
        if data.ndim == 1:
            data = np.vstack([data, data])
        elif np.mean(np.abs(data[1])) < 0.0001:
            # print("Mono source detected! Copying Left to Right...")
            data[1] = data[0].copy()

        length = data.shape[1]
        self.one_f = ofg.generate_one_f(length)

        # 2. 音量揺らぎの適用
        # そのまま掛けると音量が大きくなりすぎる場合があるので、
        # 必要に応じてここで係数を調整してもOKです。
        # 例: multiplier = (self.one_f.ifft_real_result - 1) * 0.5 + 1

        multiplier = self.one_f.ifft_real_result

        # ステレオデータの全要素に掛け算
        # shapeが違う(2行 vs 1行)ので、numpyが自動で各行に掛けてくれます
        vol_data = data * multiplier

        return vol_data

    def vid(self, be, af):
        # 1秒分だけ表示
        self.limit = int(1 * self.sr)
        fig, ax = plt.subplots(3, 1, sharex=True)

        # ステレオ対応：左チャンネル[0]だけをプロット
        ax[0].plot(be[0, : self.limit], label="Before", color="tab:blue")
        ax[1].plot(af[0, : self.limit], label="After", color="tab:orange")

        ax[2].plot(be[0, : self.limit], label="Before", color="tab:blue", alpha=1)
        ax[2].plot(af[0, : self.limit], label="After", color="tab:orange", alpha=0.5)

        for a in ax:
            # ズームは自動
            # a.set_ylim(-0.01, 0.01)
            a.legend(loc="upper right")
            a.grid(True, linestyle="--", alpha=0.5)

        plt.show()


if __name__ == "__main__":
    syn = syn_volume()

    # 1. 読み込み
    data, sr = syn.get_file_path()

    # 比較用にコピーをとっておく
    be = data.copy()

    # 2. 加工
    af = syn.syn_vol(data, sr)

    # 3. 表示
    syn.vid(be, af)

    # 再生
    # syn.file.play_from_array(af.T, sr)
