# src/dataset.py
import numpy as np
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from mido import MidiFile
import librosa
from pretty_midi import PrettyMIDI


class VelocityDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.midi_filenames = sorted(
            [f for f in os.listdir(data_dir) if f.endswith('.mid')])
        self.audio_filenames = sorted(
            [f for f in os.listdir(data_dir) if f.endswith('.wav')])

    def __len__(self):
        return len(self.midi_filenames)

    def __getitem__(self, idx):
        # Load the MIDI file
        midi_filename = os.path.join(self.data_dir, self.midi_filenames[idx])
        midi_file = PrettyMIDI(midi_filename)

        # Parse MIDI data
        midi_data = []
        note_pitches = []
        for instrument in midi_file.instruments:
            for note in instrument.notes:
                midi_data.append([note.start, note.end])
                note_pitches.append(note.pitch)
        midi_data = np.array(midi_data, dtype=np.float32)

        if len(midi_data) == 0:
            raise Exception(f"No midi data found in file {midi_filename}")

        # Load the corresponding audio file
        audio_filename = os.path.join(self.data_dir, self.audio_filenames[idx])
        audio_data, sr = librosa.load(audio_filename)
        audio_data = librosa.stft(audio_data)
        audio_data = np.abs(audio_data)

        # Convert note pitches to one-hot encoding and add a dimension
        note_pitches = np.eye(88)[np.array(note_pitches) - 21]
        # midi_data_expanded = np.repeat(midi_data[:, :, np.newaxis], 88, axis=2)
        # midi_data = midi_data_expanded * note_pitches[:, np.newaxis, :]
        midi_data = np.concatenate([midi_data, note_pitches], axis=1)

        # Extract velocity from MIDI data
        velocity = []
        for instrument in midi_file.instruments:
            for note in instrument.notes:
                velocity.append(note.velocity)
        # velocity = np.array(velocity, dtype=np.float32)
        velocity = np.eye(128)[velocity]

        if len(velocity) == 0:
            raise Exception(
                f"No note_on messages found in MIDI file {midi_filename}")

        return audio_data, midi_data, velocity


def collate_fn(batch):
    # midi notes number are not the same, pad it
    # batch: list of tuple (audio, midi, velocity)
    audio, midi, velocity = zip(*batch)

    # pad midi and velocity
    midi = pad_sequence([torch.from_numpy(m) for m in midi], batch_first=True)
    velocity = pad_sequence([torch.from_numpy(v)
                            for v in velocity], batch_first=True)

    return torch.from_numpy(np.array(audio)), midi, velocity


def test():
    # Instantiate the dataset
    data_dir = os.path.join('data', 'SMD-8s')
    dataset = VelocityDataset(data_dir)

    # Get the first sample from the dataset
    audio_data, midi_data, velocity = dataset[1]

    # Print the shapes of the tensors
    print('Audio data shape:', audio_data.shape)
    print('MIDI data shape:', midi_data.shape)
    print('Velocity shape:', velocity.shape)


if __name__ == '__main__':
    test()
