# Video Captioning System with AI Transcription

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- Ubuntu/Debian system or WSL for Windows users

### Automatic Installation

#### Clone the repository:
```bash
git clone https://github.com/Gitanshu1903/Educational-Video-Processor.git
cd video-captioning-system
```

### Manual Installation

#### Install Python dependencies:
```bash
# Install Python dependencies:
pip install faster-whisper -q
pip install moviepy==2.0.0.dev2 -q
pip install imageio==2.25.1 -q

# Install system dependencies (Ubuntu/Debian):
sudo apt-get install libcublas11 -y
sudo apt install imagemagick -y

# Configure ImageMagick permissions:
!cat /etc/ImageMagick-6/policy.xml | sed 's/none/read,write/g'> /etc/ImageMagick-6/policy.xml
```

## Quick Start

1. Place your video file in the project directory.
2. Run the script:

```bash
python src/main.py
```

3. Find your captioned video as `output.mp4` and content summary as `summary.json`.

## Configuration

### Caption Styling
```python
style = CaptionStyle(
    font_name="Helvetica",
    font_size=32,  # Auto-adjusts based on video height
    text_color="white",
    highlight_color="yellow",
    stroke_color="black",
    stroke_width=1.5,
    background_color=(64, 64, 64),
    background_opacity=0.6
)
```

### Video Processing Settings
```python
spec = VideoSpec(
    fps=24,
    codec="libx264",
    audio_codec="aac",
    threads=4,
    bitrate="8000k",
    preset="medium"
)
```
