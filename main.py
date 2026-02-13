from gui_play import gui_play as gp
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 各モジュールのインポート
from syn_volume import syn_volume
from syn_pan import syn_pan
from syn_pitch import syn_pitch
from syn_timbre import syn_timbre
from syn_reverb import syn_reverb

if __name__ == "__main__":
    # --- 1. ファイル選択と読み込み ---
    player = gp()
    file_path = player.gui_get_music()
    if not file_path:
        exit()

    print("Loading music...")
    # ステレオで読み込み (mono=False)
    data, sr = librosa.load(file_path, mono=False, sr=None, duration=180)

    # 比較用に元データを保存
    be_data = data.copy()

    # --- 2. 1/fゆらぎ加工（バケツリレー） ---

    # (1) Volume: 音量のゆらぎ
    print("Processing Volume...")
    vol_inst = syn_volume()
    data = vol_inst.syn_vol(data, sr)

    # (2) Pan: 左右のゆらぎ
    print("Processing Pan...")
    pan_inst = syn_pan()
    data = pan_inst.syn_pan(data, sr)

    # (3) Pitch: 音程/時間のゆらぎ（ワウ・フラッター）
    print("Processing Pitch...")
    pit_inst = syn_pitch()
    data = pit_inst.syn_pit(data, sr)

    # (4) Timbre: 音色のゆらぎ（フィルター）
    print("Processing Timbre...")
    tim_inst = syn_timbre()
    data = tim_inst.syn_tim(data, sr)

    # (5) Reverb: 残響のゆらぎ
    print("Processing Reverb...")
    rev_inst = syn_reverb()
    data = rev_inst.syn_rev(data, sr)

    # --- 3. 最終仕上げ ---
    # 最後に一度だけノーマライズして、絶対に音割れしないようにする
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    print("All processing done!")

    # --- 4. 再生 ---
    # グラフが出る前に音が鳴り始めます
    # (再生を止めたい場合はコメントアウトしてください)
    print("Playing...")
    player.play_from_array(data.T, sr)

    # --- 5. 可視化（ノイズチェックと1/f特性の確認） ---
    print("Generating graphs...")

    # (A) 波形の極小ズーム（ノイズ/音割れチェック）
    zoom_limit = int(0.01 * sr)  # 0.01秒
    start_sample = int(0.5 * sr)  # 0.5秒地点

    # データの長さが足りない場合の安全策
    if start_sample + zoom_limit > data.shape[1]:
        start_sample = 0

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # 波形比較
    ax[0].plot(
        be_data[0, start_sample : start_sample + zoom_limit],
        label="Original (Smooth)",
        alpha=0.8,
        color="tab:blue",
    )
    ax[0].plot(
        data[0, start_sample : start_sample + zoom_limit],
        label="Processed (Final Result)",
        alpha=0.6,
        color="tab:orange",
    )
    ax[0].set_title("Waveform Zoom (0.01s) - Check for Jagged Edges")
    ax[0].legend(loc="upper right")
    ax[0].grid(True, linestyle="--", alpha=0.5)

    # (B) スペクトル分析（1/f特性と高周波ノイズのチェック）
    f_be, Pxx_be = signal.welch(be_data[0], sr, nperseg=1024)
    f_af, Pxx_af = signal.welch(data[0], sr, nperseg=1024)

    ax[1].semilogy(f_be, Pxx_be, label="Before", color="tab:blue")
    ax[1].semilogy(f_af, Pxx_af, label="After (1/f Processed)", color="tab:orange")
    ax[1].set_title("Power Spectral Density - Check 1/f Slope")
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_ylabel("Power")
    ax[1].legend(loc="upper right")
    ax[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
