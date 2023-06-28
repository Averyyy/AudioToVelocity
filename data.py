import pretty_midi as pm
import numpy as np
import os

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchaudio
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

import librosa

from miditok import REMI  # here we choose to use REMI
from miditok.utils import get_midi_programs

# Our parameters
pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': False, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250)}  # (min, max)
special_tokens = ["PAD", "BOS", "EOS", "MASK"]

# Creates the tokenizer and loads a MIDI
tokenizer = REMI(pitch_range, beat_res, nb_velocities,
                 additional_tokens, special_tokens)


D_DIM = 256


class SMDDataset(Dataset):

    def __init__(self, data_dir, sr=22050):
        self.data_dir = data_dir
        self.sr = sr
        self.audios, self.midis, self.labels = [], [], []
        self.loadData()

        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        audio = torch.tensor(self.audios[idx])
        label = torch.tensor(self.labels[idx])

        max_audio_length = 64000
        max_label_length = 64000

        # Pad or truncate audio
        if audio.shape[1] < max_audio_length:
            padding_size = max_audio_length - audio.shape[1]
            audio = F.pad(audio, (0, padding_size))
        else:
            audio = audio[:, :max_audio_length]

        # Pad or truncate label
        if label.shape[1] < max_label_length:
            padding_size = max_label_length - label.shape[1]
            label = F.pad(label, (0, padding_size))
        else:
            label = label[:, :max_label_length]

        # audio = audio.permute(1, 0)

        return audio, label

    def loadData(self):

        print("-----------Start loading file------------")

        files = os.listdir(self.data_dir)

        for i, file in enumerate(files):
            filename, typ = file.split('.')

            if typ == 'mid':

                print("Loading file:", filename)

                # midi = pm.PrettyMIDI(os.path.join(self.data_dir, file)).instruments[0].notes

                audioFileName = filename + '.mp3'

                # audioInMel = np.load(os.path.join(self.data_dir, audioFileName))
                # S_dB = librosa.power_to_db(audioInMel, ref=1e-4)

                wave, sr = librosa.load(
                    os.path.join(self.data_dir, audioFileName))
                S = np.abs(librosa.stft(wave, n_fft=2046,
                           hop_length=512, center=False))
                S_dB = librosa.amplitude_to_db(S, ref=1e-4)

                self.audios.append(S_dB)

                midi_tok = tokenizer(os.path.join(self.data_dir, file))
                self.labels.append(midi_tok)

                # for note in midi:
                #     self.midis.append([note.pitch])
                #     self.labels.append([note.velocity])

                #     start = int((note.start * self.sr) / 512)
                #     end = int((note.end * self.sr) / 512)

                #     dur = end - start
                #     if dur < D_DIM:
                #         pad = torch.zeros(1025, D_DIM - dur)

                #         mel = S_dB[:, start: end]

                #         # add paddings
                #         mel = np.concatenate((mel, pad), 1)

                #     else:
                #         end = start + D_DIM
                #         mel = S_dB[:, start: end]

                #     self.audios.append(mel)

                # if i == 3 * 0:
                #     break

            else:
                continue

        print("-----------Stop loading file------------")

        # return midis, audios, labels
