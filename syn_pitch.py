from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa
import matplotlib.pyplot as plt
import numpy as np


class syn_pitch:
    def get_file_path(self):
        self.file = gp()
        self.file_path = self.file.gui_get_music()
        # ここでは読み込むだけ
        self.data, self.sr = librosa.load(
            self.file_path, mono=False, sr=None, duration=180
        )
        return self.data, self.sr

    def syn_pit(self, data, sr):
        # 1. 安全装置：モノラル対策
        if data.ndim == 1:
            data = np.vstack([data, data])
        elif np.mean(np.abs(data[1])) < 0.0001:
            print("Mono source detected! Copying Left to Right...")
            data[1] = data[0].copy()

        # 2. 処理開始
        length = data.shape[1]
        self.one_f = ofg.generate_one_f(length)

        # テープ伸び係数（0.05くらいが適量）
        fluctuation = (self.one_f.ifft_real_result - 1) * 0.05

        # 時間の歪みマップ作成
        speed_map = 1.0 + fluctuation
        dirty_time_index = np.cumsum(speed_map)

        # 尺合わせ（元の曲の長さに強制的に合わせる）
        dirty_time_index = dirty_time_index / dirty_time_index[-1] * (length - 1)

        # リサンプリング（補間）実行
        original_index = np.arange(length)

        # 左右それぞれ加工
        data[0] = np.interp(dirty_time_index, original_index, data[0])
        data[1] = np.interp(dirty_time_index, original_index, data[1])

        # 加工済みデータを返す（バケツリレー用）
        return data

    def vid(self, lf, ri):
        # Pitchは変化が見えにくいので、長めに10秒表示
        self.limit = int(10 * self.sr)
        fig, ax = plt.subplots(3, 1, sharex=True)

        ax[0].plot(lf[: self.limit], label="Left (Pitch Mod)", color="tab:blue")
        ax[1].plot(ri[: self.limit], label="Right (Pitch Mod)", color="tab:orange")

        # 重ねて表示（ズレが見えるかも）
        ax[2].plot(lf[: self.limit], label="Left", color="tab:blue", alpha=0.7)
        ax[2].plot(ri[: self.limit], label="Right", color="tab:orange", alpha=0.7)

        for a in ax:
            a.legend(loc="upper right")
            a.grid(True, linestyle="--", alpha=0.5)

        plt.show()


if __name__ == "__main__":
    syn = syn_pitch()
    # 1. 読み込み
    data, sr = syn.get_file_path()

    # 2. 加工（戻り値で上書き）
    data = syn.syn_pit(data, sr)

    # 3. 表示
    syn.vid(data[0], data[1])

    # 再生確認したければここを開放
    # syn.file.play_from_array(data.T, sr)
