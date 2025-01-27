# Audio Transcription and Speaker Diarization

This project provides tools for audio transcription and speaker diarization using Whisper and Pyannote. It includes functionality for basic transcription, audio analysis, and speaker identification in audio files.

## Features

- Audio transcription using OpenAI's Whisper
- Speaker diarization using Pyannote
- Audio analysis including:
  - Language detection
  - Log-mel spectrogram visualization
  - Audio segment extraction
- Complete transcription pipeline with speaker identification

## Project Structure

```
.
├── Audio/                     # Directory for input audio files
├── Transcription/            # Directory for output transcriptions
├── diarize_transcribe.ipynb  # Notebook for transcription with speaker diarization
├── check_transcribe.ipynb    # Notebook for audio analysis and transcription
├── pyproject.toml           # Project dependencies and metadata
├── uv.lock                  # UV lock file for reproducible builds
├── .env                     # Environment variables (create this file)
└── README.md
```

## Prerequisites

- Python 3.12
- FFmpeg (for audio processing)
- CUDA-capable GPU (recommended for faster processing but works without to)
- UV package manager

## Installation

1. Install UV (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv
   # On Windows
   .venv\Scripts\activate
   # On Unix or MacOS
   source .venv/bin/activate
   ```

4. Install dependencies using UV:
   ```bash
   uv pip install --requirements pyproject.toml
   ```

5. Set up environment variables:
   Create a `.env` file in the root directory with:
   ```
   HF_TOKEN=your_hugging_face_token
   ```
   Get your Hugging Face token from: https://huggingface.co/settings/tokens

## Usage

### Basic Transcription and Analysis

Use `check_transcribe.ipynb` for:
- Basic audio transcription
- Language detection
- Log-mel spectrogram visualization
- Audio segment extraction

Example:
```python
# Process an audio file
results = process_audio('Audio/your_audio.wav', 
                       output_dir='Transcription',
                       segment=('02:00', '03:00'))  # Optional segment
```

### Speaker Diarization

Use `diarize_transcribe.ipynb` for:
- Full transcription with speaker identification
- Timeline-based speaker segmentation
- Combined transcription and speaker information

Example:
```python
# Transcribe with speaker diarization
AUDIO_FILE = "Audio/your_audio.wav"
result = whisper_model.transcribe(AUDIO_FILE)
diarization = pipeline(AUDIO_FILE)
```

## Input/Output

### Input
- Place your audio files in the `Audio/` directory
- Supported formats: WAV files (recommended)
- Other formats may need conversion using FFmpeg

### Output
Transcriptions and analysis results are saved in the `Transcription/` directory:
- `transcription.txt`: Raw transcription
- `transcription_with_speakers.txt`: Transcription with speaker identification
- Additional analysis files (spectrograms, etc.)

## Dependencies

Dependencies are managed using UV and specified in `pyproject.toml`. Key dependencies include:
- ipykernel
- ipywidgets
- llvmlite
- numba
- openai-whisper
- pyannote-audio
- python-dotenv
- torch
- tqdm

For exact versions, refer to `uv.lock` file.

## Models

### Whisper
Available models at time of writing (see also [Whisper repo](https://github.com/openai/whisper)):
- tiny.en, tiny
- base.en, base
- small.en, small
- medium.en, medium
- large-v1, large-v2, large-v3
- large, large-v3-turbo

Default: `large-v3-turbo` (best balance of speed and accuracy)

### Pyannote
Using `pyannote/speaker-diarization-3.1` for speaker diarization

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [UV Package Manager](https://github.com/astral-sh/uv)
