from gui_play import gui_play as gp
import librosa  # 読み込み用

# 【修正1】クラスを直接指名してインポートする
from syn_volume import syn_vol
from syn_pan import syn_pan
from syn_pitch import syn_pit
from syn_timbre import syn_tim
from syn_reverb import syn_rev

if __name__ == "__main__":
    # 1. プレイヤー準備 & ファイル選択
    player = gp()
    file_path = player.gui_get_music()

    if not file_path:
        print("No file selected.")
        exit()

    # 2. ここで一回だけ読み込む（これが最初の素材）
    print("Loading music... (Stereo)")
    # mono=Falseを忘れずに！
    data, sr = librosa.load(file_path, mono=False, sr=None, duration=180)

    # -------------------------------------------------
    # 3. 究極のバケツリレー開始
    # -------------------------------------------------

    # 第1走者：Volume（音量）
    print("Applying 1/f Volume...")
    vol = syn_vol()
    data = vol.syn_vol(data, sr)  # dataを上書きしていく

    # 第2走者：Pan（左右）
    print("Applying 1/f Pan...")
    pan = syn_pan()
    data = pan.syn_pan(data, sr)

    # 第3走者：Pitch（音程・テープ伸び）
    print("Applying 1/f Pitch...")
    pit = syn_pitch()
    data = pit.syn_pitch(data, sr)

    # 第4走者：Timbre（音色・こもり）
    print("Applying 1/f Timbre...")
    tim = syn_timbre()
    data = tim.syn_timbre(data, sr)

    # アンカー：Reverb（空間・残響）
    # ※重いので最後にやるのがセオリー
    print("Applying 1/f Reverb... (This may take time)")
    rev = syn_reverb()
    data = rev.syn_reverb(data, sr)

    # -------------------------------------------------
    # 4. 再生
    # -------------------------------------------------
    print("Playing generated chill music...")
    # 再生時は転置(.T)が必要
    player.play_from_array(data.T, sr)
