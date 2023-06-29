# Music Transformer for Piano Performance

This repository contains the implementation of a Music Transformer model that takes piano recordings as input and outputs MIDI parameters such as pitch and velocity.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Music Transformer is a model that aims to generate MIDI parameters from piano recordings. It uses a Transformer architecture, which is known for its effectiveness in sequence-to-sequence tasks. This can be useful for musicians, composers, and researchers who are interested in music generation and analysis.

## Installation

Before running the code, you need to install the required libraries. You can install them using `pip`:

```sh
pip install torch torchaudio librosa miditok
```

## Dataset

The dataset used is SMD (Sample Music Dataset). It consists of piano recordings in various formats including `.mid`, `.mp3`, and `.npy`. Each recording is associated with MIDI parameters such as pitch and velocity.

Example of files in the dataset:

```
-Bach_BWV849-01_001_20090916-SMD.mid
-Bach_BWV849-01_001_20090916-SMD.npy
-Bach_BWV849-01_001_20090916-SMD.mp3
-Bartok.....mid
...
```

## Model Architecture

The model is based on the Transformer architecture, which consists of an encoder and a decoder. The encoder processes the input piano recordings, and the decoder generates the MIDI parameters.

The encoder is composed of multiple layers, where each layer contains multi-head self-attention mechanisms and feed-forward neural networks. The decoder is similar to the encoder but also includes additional multi-head attention mechanisms that attend to the output of the encoder stack.

## Usage

To train the model, run the main script:

```sh
python main.py
```

This will train the model using the SMD dataset. The trained model will be saved as `piano_model.pth`.

To use the trained model for generating MIDI parameters from new piano recordings, you can load the model and pass the recordings through it:

```python
model = MusicTransformer(...)
model.load_state_dict(torch.load('piano_model.pth'))
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or create an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
