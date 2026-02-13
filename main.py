from gui_play import gui_play as gp
import librosa

# インポート（ここは現状のファイル名とクラス名に合わせています）
from syn_volume import syn_volume
from syn_pan import syn_pan
from syn_pitch import syn_pitch
from syn_timbre import syn_timbre
from syn_reverb import syn_reverb

if __name__ == "__main__":
    player = gp()
    file_path = player.gui_get_music()
    if not file_path:
        exit()

    print("Loading music...")
    data, sr = librosa.load(file_path, mono=False, sr=None, duration=180)

    # 1. Volume (メソッド名が syn_vol のはず)
    vol_inst = syn_volume()
    data = vol_inst.syn_vol(data, sr)

    # 2. Pan (メソッド名が syn_pan のはず)
    pan_inst = syn_pan()
    data = pan_inst.syn_pan(data, sr)

    # 3. Pitch
    # 【修正箇所】エラー通り、メソッド名を syn_pit に合わせる！
    pit_inst = syn_pitch()
    data = pit_inst.syn_pit(data, sr)  # syn_pitch ではなく syn_pit

    # 4. Timbre
    # 【修正箇所】ここもおそらく syn_tim になっているはず！
    tim_inst = syn_timbre()
    data = tim_inst.syn_tim(data, sr)  # syn_timbre ではなく syn_tim

    # 5. Reverb
    # 【修正箇所】ここもおそらく syn_rev になっているはず！
    rev_inst = syn_reverb()
    data = rev_inst.syn_rev(data, sr)  # syn_reverb ではなく syn_rev

    print("Playing...")
    player.play_from_array(data.T, sr)
