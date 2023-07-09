import os
import librosa
import pretty_midi
import numpy as np
import soundfile as sf


def parse_files(data_dir, output_dir, segment_length=8, window_hop_length=4):
    for filename in os.listdir(data_dir):
        if filename.endswith('.mp3'):
            audio_file = os.path.join(data_dir, filename)
            midi_file = audio_file.replace('.mp3', '.mid')

            # Load the audio file using librosa
            y, sr = librosa.load(audio_file)

            # Load the MIDI file using pretty_midi
            midi_data = pretty_midi.PrettyMIDI(midi_file)

            # Calculate the total number of samples and samples per segment
            total_samples = len(y)
            samples_per_segment = int(segment_length * sr)
            hop_samples = int(window_hop_length * sr)
            num_segments = int(
                np.ceil((total_samples - samples_per_segment) / hop_samples)) + 1

            for i in range(num_segments):
                # Calculate the start and end sample indices of the segment
                start_sample = i * hop_samples
                end_sample = start_sample + samples_per_segment

                # If the end sample exceeds the total number of samples, start from the last sample
                if end_sample > total_samples:
                    end_sample = total_samples
                    start_sample = end_sample - samples_per_segment

                # Convert sample indices to time in seconds
                start_time = start_sample / sr
                end_time = end_sample / sr

                # Cut audio segment
                audio_segment = y[start_sample:end_sample]

                # Only save the segment if it is not empty
                if len(audio_segment) > 0:
                    # Cut MIDI segment
                    midi_segment = []
                    for instrument in midi_data.instruments:
                        for note in instrument.notes:
                            # If the note is in the current segment
                            if start_time <= note.start < end_time or start_time < note.end <= end_time:
                                midi_segment.append(note)

                    # Save segmented files
                    # Construct the output filename
                    output_filename = filename.replace('.mp3', f'_{i}.wav')
                    # Construct the full path of the output file
                    output_file = os.path.join(output_dir, output_filename)

                    # Save audio segment as a .wav file
                    sf.write(output_file, audio_segment, sr)

                    # Save MIDI segment as a .mid file
                    midi_segment_data = pretty_midi.PrettyMIDI()
                    new_instrument = pretty_midi.Instrument(program=0)
                    new_instrument.notes = midi_segment
                    midi_segment_data.instruments.append(new_instrument)
                    midi_segment_data.write(
                        output_file.replace('.wav', '.mid'))


if __name__ == "__main__":
    data_dir = os.path.join('data', 'SMD')
    output_dir = os.path.join('data', 'SMD-3s')
    os.makedirs(output_dir, exist_ok=True)
    parse_files(data_dir, output_dir)
