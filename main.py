import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import threading
import sounddevice as sd

# Tkinterにグラフを埋め込むための魔法の呪文
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# 既存モジュールのインポート
from gui_play import gui_play as gp
from syn_volume import syn_volume
from syn_pan import syn_pan
from syn_pitch import syn_pitch
from syn_timbre import syn_timbre
from syn_reverb import syn_reverb


class MusicOneFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("1/f Fluctuation Generator v2.0 (Stable)")
        self.root.geometry("600x500")

        # グラフ表示用のサブウィンドウ管理
        self.graph_window = None

        # 変数
        self.file_path = tk.StringVar()
        self.status = tk.StringVar(value="Ready")

        # パラメータ (Default 1.0)
        self.depth_vol = tk.DoubleVar(value=1.0)
        self.depth_pan = tk.DoubleVar(value=1.0)
        self.depth_pit = tk.DoubleVar(value=1.0)
        self.depth_tim = tk.DoubleVar(value=1.0)
        self.depth_rev = tk.DoubleVar(value=1.0)

        self._create_widgets()

    def _create_widgets(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        # 1. ファイル選択
        frame_file = ttk.LabelFrame(main_frame, text="Source File", padding=10)
        frame_file.pack(fill="x", pady=5)

        btn_file = ttk.Button(
            frame_file, text="Select Music", command=self._select_file
        )
        btn_file.pack(side="left")
        lbl_path = ttk.Label(
            frame_file, textvariable=self.file_path, font=("Consolas", 8)
        )
        lbl_path.pack(side="left", padx=5)

        # 2. パラメータスライダー
        frame_param = ttk.LabelFrame(
            main_frame,
            text="Fluctuation Depth (0.2=Natural, 1.0=Standard, 3.0=Chaos)",
            padding=10,
        )
        frame_param.pack(fill="x", pady=5)

        self._add_slider(frame_param, "Volume", self.depth_vol)
        self._add_slider(frame_param, "Pan", self.depth_pan)
        self._add_slider(frame_param, "Pitch (Danger)", self.depth_pit)
        self._add_slider(frame_param, "Timbre", self.depth_tim)
        self._add_slider(frame_param, "Reverb", self.depth_rev)

        # 3. 実行・停止ボタン
        frame_action = ttk.Frame(main_frame, padding=10)
        frame_action.pack(fill="x", pady=5)

        btn_run = ttk.Button(
            frame_action, text="GENERATE & PLAY", command=self._start_processing_thread
        )
        btn_run.pack(side="left", fill="x", expand=True, padx=5, ipady=5)

        btn_stop = ttk.Button(frame_action, text="STOP", command=self._stop_playback)
        btn_stop.pack(side="right", fill="x", expand=True, padx=5, ipady=5)

        # ステータスバー
        lbl_status = ttk.Label(self.root, textvariable=self.status, relief="sunken")
        lbl_status.pack(side="bottom", fill="x")

    def _add_slider(self, parent, label, variable):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=2)
        ttk.Label(f, text=label, width=15).pack(side="left")
        s = ttk.Scale(f, from_=0.0, to=3.0, variable=variable, orient="horizontal")
        s.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(f, textvariable=variable, width=5).pack(side="right")

    def _select_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a")]
        )
        if path:
            self.file_path.set(path)

    def _stop_playback(self):
        sd.stop()
        self.status.set("Playback Stopped.")

    def _start_processing_thread(self):
        """計算スレッドの開始"""
        if not self.file_path.get():
            messagebox.showwarning("Warning", "ファイルを選択してください！")
            return

        self.status.set("Processing... (Please wait)")
        # 計算は重いので別スレッドで実行
        threading.Thread(target=self._process_logic, daemon=True).start()

    def _process_logic(self):
        """裏方で行う重い計算処理"""
        try:
            path = self.file_path.get()

            # 1. 読み込み
            data, sr = librosa.load(path, mono=False, sr=None, duration=180)
            if data.ndim == 1:
                data = np.vstack([data, data])

            be_data = data.copy()

            # 2. バケツリレー加工 (GUIパラメータ注入)
            # 各Depthを取得（スレッド内での参照は安全）
            d_vol = self.depth_vol.get()
            d_pan = self.depth_pan.get()
            d_pit = self.depth_pit.get()
            d_tim = self.depth_tim.get()
            d_rev = self.depth_rev.get()

            # (1) Volume
            if d_vol > 0:
                inst = syn_volume()
                processed = inst.syn_vol(data, sr)
                data = data + (processed - data) * d_vol

            # (2) Pan
            if d_pan > 0:
                inst = syn_pan()
                processed = inst.syn_pan(data, sr)
                data = data + (processed - data) * d_pan

            # (3) Pitch (ブレンド不可なので直接代入か、弱めるならmix)
            if d_pit > 0:
                inst = syn_pitch()
                processed = inst.syn_pit(data, sr)
                # Pitchはブレンドすると二重になるので、本来は内部パラメータを変えるべきだが
                # 今回は 0 か 1 かの挙動に近い形になる
                data = processed

            # (4) Timbre
            if d_tim > 0:
                inst = syn_timbre()
                processed = inst.syn_tim(data, sr)
                data = data + (processed - data) * d_tim

            # (5) Reverb
            if d_rev > 0:
                inst = syn_reverb()
                processed = inst.syn_rev(data, sr)
                data = data + (processed - data) * d_rev

            # 最終ノーマライズ
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val

            # 計算完了！ -> メインスレッドに「描画と再生」を依頼する
            # root.after(0, 関数, 引数...) を使うと、安全にメインスレッドで実行できる
            self.root.after(0, self._finish_processing, be_data, data, sr)

        except Exception as e:
            # エラー時もメインスレッドでメッセージを出す
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            print(e)

    def _finish_processing(self, be_data, data, sr):
        """メインスレッドで行う描画と再生"""
        self.status.set("Playing...")

        # 1. 再生開始 (非同期)
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        sd.play(data.T, sr)

        # 2. グラフ描画 (安全な別ウィンドウ)
        self._show_stable_graph(be_data, data, sr)

    def _show_stable_graph(self, be_data, data, sr):
        """Tkinter Toplevelを使った安全なグラフ表示"""

        # 既にグラフウィンドウが開いていたら閉じてリセット
        if self.graph_window is not None and self.graph_window.winfo_exists():
            self.graph_window.destroy()

        # 新しいウィンドウを作成
        self.graph_window = tk.Toplevel(self.root)
        self.graph_window.title("Analysis Result")
        self.graph_window.geometry("800x600")

        # MatplotlibのFigureを作成
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))

        # (A) Waveform
        zoom_limit = int(0.01 * sr)
        start_sample = int(0.5 * sr)
        if start_sample + zoom_limit > data.shape[1]:
            start_sample = 0

        ax[0].plot(
            be_data[0, start_sample : start_sample + zoom_limit],
            alpha=0.8,
            label="Original",
        )
        ax[0].plot(
            data[0, start_sample : start_sample + zoom_limit],
            alpha=0.6,
            label="Processed",
        )
        ax[0].set_title("Waveform Zoom (0.01s)")
        ax[0].legend(loc="upper right")

        # (B) PSD (Ideal 1/f)
        f_be, Pxx_be = signal.welch(be_data[0], sr, nperseg=1024)
        f_af, Pxx_af = signal.welch(data[0], sr, nperseg=1024)

        # 理想直線の計算
        f_ideal = f_af.copy()
        f_ideal[0] = 1e-10
        P_ideal = 1.0 / f_ideal
        ref_mask = (f_af > 100) & (f_af < 1000)
        if np.sum(ref_mask) > 0:
            scale_factor = np.mean(Pxx_af[ref_mask]) / np.mean(P_ideal[ref_mask])
            P_ideal = P_ideal * scale_factor

        ax[1].loglog(f_be, Pxx_be, label="Original", alpha=0.5)
        ax[1].loglog(f_af, Pxx_af, label="Processed")
        ax[1].loglog(f_ideal, P_ideal, label="Ideal 1/f", linestyle="--", color="green")
        ax[1].set_title("Spectral Density")
        ax[1].legend(loc="upper right")
        ax[1].grid(True, which="both", linestyle="--")

        plt.tight_layout()

        # FigureをTkinterウィンドウに埋め込む
        canvas = FigureCanvasTkAgg(fig, master=self.graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # ツールバー（ズームとか保存ができるやつ）もつける
        toolbar = NavigationToolbar2Tk(canvas, self.graph_window)
        toolbar.update()
        canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    # 終了時にスレッドも道連れにする設定
    app = MusicOneFApp(root)
    root.mainloop()
