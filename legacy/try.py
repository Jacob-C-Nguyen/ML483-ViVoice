import soundfile as sf
import os

for fname in os.listdir("dataset"):
    if fname.endswith(".wav"):
        fpath = os.path.join("dataset", fname)
        try:
            data, sr = sf.read(fpath)
            if len(data) == 0:
                print(f"EMPTY: {fname}")
            elif sr != 24000:
                print(f"WRONG SR ({sr}Hz): {fname}")
            else:
                print(f"OK: {fname}")
        except Exception as e:
            print(f"BAD: {fname} - {e}")