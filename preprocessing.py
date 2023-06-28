import librosa
import numpy as np
import os

N_MELS = 128
N_FFT = 2048


def convertWaveToMelSpectrogram():

    data_dir = '../../dataset/SMD/'

    files = os.listdir(data_dir)

    for file in files:
        filename, typ = file.split('.')

        print("Loading file:", filename)

        if typ == 'mid':
            continue

        else:

            wave, sr = librosa.load(os.path.join(data_dir, file))

            S = librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=N_MELS, n_fft=N_FFT)

            file_save = filename + '.npy'
            np.save(os.path.join(data_dir, file_save), S)


convertWaveToMelSpectrogram()
