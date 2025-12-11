import whisper


model = whisper.load_model("base")

MyMes1 = model.transcribe(".\Sounds\MyMes1.wav")
MyMes1_file = open(".\Results\MyMes1.txt", "w")
MyMes1_file.write(MyMes1["text"])
MyMes1_file.close()

MyMes2 = model.transcribe(".\Sounds\MyMes2.wav")
MyMes2_file = open(".\Results\MyMes2.txt", "w")
MyMes2_file.write(MyMes2["text"])
MyMes2_file.close()

MyMes3 = model.transcribe(".\Sounds\MyMes3.wav")
MyMes3_file = open(".\Results\MyMes3.txt", "w")
MyMes3_file.write(MyMes3["text"])
MyMes3_file.close()

MyMes4 = model.transcribe(".\Sounds\MyMes4.wav")
MyMes4_file = open(".\Results\MyMes4.txt", "w")
MyMes4_file.write(MyMes4["text"])
MyMes4_file.close()

Peas = model.transcribe(".\Sounds\PeasantMes.wav")
Peas_file = open(".\Results\PeasantMes.txt", "w")
Peas_file.write(Peas["text"])
Peas_file.close()

Rom = model.transcribe(".\Sounds\RomMes.wav")
Rom_file = open(".\Results\RomMes.txt", "w")
Rom_file.write(Rom["text"])
Rom_file.close()

Mom = model.transcribe(".\Sounds\MomMes.wav")
Mom_file = open(".\Results\MomMes.txt", "w")
Mom_file.write(Mom["text"])
Mom_file.close()

Kate = model.transcribe(".\Sounds\KatyaMes.wav")
Kate_file = open(".\Results\KatyaMes.txt", "w")
Kate_file.write(Kate["text"])
Kate_file.close()

Conf1 = model.transcribe(".\Sounds\Confess1.wav")
Conf1_file = open(".\Results\Confess1.txt", "w")
Conf1_file.write(Conf1["text"])
Conf1_file.close()

Conf2 = model.transcribe(".\Sounds\Confess2.wav")
Conf2_file = open(".\Results\Confess2.txt", "w")
Conf2_file.write(Conf2["text"])
Conf2_file.close()

# negr = model.transcribe(".\Sounds\egr.wav")
# negr_file = open(".\Results\egr.txt", "w")
# negr_file.write(negr["text"])
# negr_file.close()

mat = model.transcribe(".\Sounds\mat.wav")
mat_file = open(".\Results\mat.txt", "w")
mat_file.write(mat["text"])
mat_file.close()

chu = model.transcribe("Sounds\Churchill.mp3")
chu_file = open(".\Results\Churchill.txt", "w")
chu_file.write(chu["text"])
chu_file.close()

wake = model.transcribe("Sounds\Wakeup.ogg")
wake_file = open(".\Results\Wakeup.txt", "w")
wake_file.write(wake["text"])
wake_file.close()

# world = model.transcribe("Sounds\TheWorld.mp3")
# world_file = open(".\Results\TheWorld.txt", "w")
# world_file.write(world["text"])
# world_file.close()

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = r".\Sounds\Confess2.wav" 
y, sr = librosa.load(audio_path, sr=16000)

print("Sample rate:", sr)
print("Length (sec):", len(y) / sr)

S = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=1024,
    hop_length=256,
    n_mels=128,
    fmin=20,
    fmax=8000
)

S_db = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    S_db,
    sr=sr,
    hop_length=256,
    x_axis='time',
    y_axis='mel'
)
plt.colorbar(format='%+2.0f dB')
plt.title("Mel-Spectrogram of Confess2.wav")
plt.tight_layout()
plt.show()
