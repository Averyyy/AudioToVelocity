# Music Transformer for Piano Performance

This repository contains the implementation of a Music Transformer model that takes piano recordings as input and outputs MIDI parameters such as pitch and velocity.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or create an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

positional encoding
time normalize
midi padding mask
loss ignore index

velocity bin
分别 embed start, end, pitch(one hot)
data augmentation(audio(torch_audio), velocity(add noise), time(add noise))
