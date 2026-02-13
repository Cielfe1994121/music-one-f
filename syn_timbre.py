from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal  # フィルター用


class syn_timbre:
    def get_file_path(self):
        self.file = gp()
        self.file_path = self.file.gui_get_music()
        self.data, self.sr = librosa.load(
            self.file_path, mono=False, sr=None, duration=180
        )
        return self.data, self.sr

    def syn_timbre(self, data, sr):
        # 1. 安全装置：モノラル対策
        if data.ndim == 1:
            data = np.vstack([data, data])
        elif np.mean(np.abs(data[1])) < 0.0001:
            print("Mono source detected! Copying Left to Right...")
            data[1] = data[0].copy()

        length = data.shape[1]

        # 2. フィルター作成（こもった音を作る準備）
        # バターワースフィルタ（ローパス）
        nyquist = sr / 2
        cutoff = 1000 / nyquist
        b, a = signal.butter(4, cutoff, btype="low")

        # フィルター適用
        muffled_data = np.zeros_like(data)
        muffled_data[0] = signal.lfilter(b, a, data[0])
        muffled_data[1] = signal.lfilter(b, a, data[1])

        # 3. 1/fゆらぎ係数を作る
        self.one_f = ofg.generate_one_f(length)

        # ブレンド率（0.0〜1.0）の計算
        # 揺れ幅を調整（* 5.0 くらいがダイナミックで良い）
        mix_ratio = (self.one_f.ifft_real_result - 1) * 5.0
        mix_ratio = np.clip(mix_ratio + 0.5, 0.0, 1.0)

        # 4. ブレンド（Original vs Muffled）
        # 元のデータ(data)は書き換えずに、新しい配列(processed_data)を作って返す
        processed_data = data * (1 - mix_ratio) + muffled_data * mix_ratio

        return processed_data

    def vid(self, be, af):
        # 1秒分だけ表示
        self.limit = int(1 * self.sr)
        fig, ax = plt.subplots(3, 1, sharex=True)

        # 左チャンネルだけを比較表示
        ax[0].plot(be[0, : self.limit], label="Before", color="tab:blue")
        ax[1].plot(af[0, : self.limit], label="After (Muffled Mix)", color="tab:orange")

        # 重ね合わせ
        ax[2].plot(be[0, : self.limit], label="Before", color="tab:blue", alpha=1)
        ax[2].plot(af[0, : self.limit], label="After", color="tab:orange", alpha=0.5)

        for a in ax:
            # ズームは自動（変化が見やすいように固定解除）
            # a.set_ylim(-1.0, 1.0)
            a.legend(loc="upper right")
            a.grid(True, linestyle="--", alpha=0.5)

        plt.show()


if __name__ == "__main__":
    syn = syn_timbre()

    # 1. 読み込み
    data, sr = syn.get_file_path()

    # 元データを比較用に取っておく（加工で変わるかもしれないのでcopy推奨）
    be = data.copy()

    # 2. 加工
    # 戻り値（加工後のデータ）を受け取る
    af = syn.syn_timbre(data, sr)

    # 3. 表示
    syn.vid(be, af)

    # 再生確認用
    # syn.file.play_from_array(af.T, sr)
