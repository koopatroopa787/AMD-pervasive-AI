# Streamline: AI-Powered Multi-Language Video Summarizer and Upscaler

# Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/koopatroopa787">
        <img src="https://github.com/koopatroopa787.png" width="100px;" alt="Profile Picture"/><br />
        <sub><b>Kanishk Kumar Sachan</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Divine-pro">
        <img src="https://github.com/Divine-pro.png" width="100px;" alt="Profile Picture"/><br />
        <sub><b>Iltimas Kabir</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/darryll-git">
        <img src="https://github.com/darryll-git.png" width="100px;" alt="Profile Picture"/><br />
        <sub><b>Darryll Fonseca</b></sub>
      </a>
    </td>
  </tr>
</table>


## Project Description
Streamline is an advanced AI-powered tool that enhances and simplifies video content consumption. It provides automated video summarization, multi-language translation, and upscaling, making content accessible and high-quality for a global audience.

## Directory Structure
```plaintext
/home/ubuntu/streamlit
├── app.py
├── output_directory
│   ├── final_summary.txt
│   ├── key_points.txt
│   ├── short_summary.txt
│   ├── translated_key_points.wav
│   ├── translated_short_summary.wav
│   └── translated_summary.wav
├── requirements.txt
├── something.py
├── stream.py
├── streamlit
│   ├── audio.mp3
│   ├── formatted_audio.wav
│   ├── upscaled_video.mp4
│   ├── upscaled_video_h264.mp4
│   └── video.mp4
├── utils
│   ├── __pycache__
│   ├── audio.py
│   ├── summarization.py
│   ├── transcription.py
│   ├── translation.py
│   ├── upscaling.py
│   ├── video.py
│   └── youtube.py
├── video.mp4
└── vosk-model-en-us-0.22-lgraph
    ├── README
    ├── am
    ├── conf
    ├── graph
    └── ivector
```
# Setup Instructions
## Prerequisites
```plaintext
Python 3.8 or higher
pip
```
# Installation
## Clone the repository:
```plaintext
https://github.com/koopatroopa787/AMD-pervasive-AI.git
cd AMD-pervasive-AI
```

```plaintext
pip install -r requirements.txt
```
## Download the Vosk model:

```plaintext
mkdir vosk-model-en-us-0.22-lgraph
cd vosk-model-en-us-0.22-lgraph
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip
unzip vosk-model-en-us-0.22-lgraph.zip
rm vosk-model-en-us-0.22-lgraph.zip
cd ..
```
# Running the Application
## To start the Streamlit application, run:
```plaintext
streamlit run app.py
```
