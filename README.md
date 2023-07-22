# Music Transformer for Piano Performance

This repository contains the implementation of a Music Transformer model that takes piano recordings as input and outputs MIDI parameters such as pitch and velocity.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or create an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

positional encoding dev done
time normalize dev done
midi padding mask done  
loss ignore index done

velocity bin pending
分别 embed start, end, pitch(one hot) dev done
data augmentation(audio(torch_audio), velocity(add noise), time(add noise)) pending
