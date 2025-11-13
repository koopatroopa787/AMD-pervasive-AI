# 🎬 AI Video Processor (Streamline Enhanced)

An enterprise-grade video processing application combining multiple AI technologies for video summarization, transcription, translation, and upscaling. Now with **multi-model/API support** and significantly improved code quality!

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 👥 Contributors

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

## 📖 Project Description

AI Video Processor (formerly Streamline) is an advanced AI-powered tool that enhances and simplifies video content consumption. It provides automated video summarization, multi-language translation, and upscaling, making content accessible and high-quality for a global audience.

## ✨ What's New in Enhanced Version

### 🚀 Major Improvements

1. **🔌 Multi-Model/API Support** - Now supports 5+ AI providers!
   - 🤗 Hugging Face (Mistral, Llama, Phi-3, etc.)
   - 🟢 OpenAI (GPT-4, GPT-3.5-turbo)
   - 🔵 Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
   - 🦙 Ollama (Local deployment)
   - ⚡ Groq (Ultra-fast inference)

2. **🔒 Enhanced Security**
   - Fixed critical shell injection vulnerabilities
   - Secure subprocess handling (no more `shell=True`)
   - Input validation and sanitization
   - Path traversal protection

3. **⚙️ Configuration System**
   - Centralized configuration management
   - Environment variable support (.env files)
   - Runtime validation
   - No more hardcoded paths!

4. **📊 Professional Logging & Error Handling**
   - Colored console output
   - Structured logging
   - Detailed error messages
   - Debug mode support

5. **🎨 Improved UI/UX**
   - Modern, responsive design
   - Real-time progress indicators
   - Tabbed result display
   - System information panel
   - Better visual feedback

6. **⚡ Performance Optimizations**
   - Model caching
   - Lazy loading
   - Configurable parallelization
   - GPU acceleration support

7. **📚 Code Quality**
   - Full type hints
   - Comprehensive docstrings
   - Better error messages
   - Modular architecture

## 📁 Directory Structure

```plaintext
AMD-pervasive-AI/
├── app.py                      # Main Streamlit application (enhanced!)
├── app_legacy.py              # Original version (backup)
├── config.py                   # Configuration management (NEW!)
├── requirements.txt            # Python dependencies (updated)
├── .env.example               # Environment configuration template (NEW!)
├── .env                       # Your configuration (create this)
│
├── utils/                     # Utility modules (all improved!)
│   ├── audio.py              # Audio processing (secure)
│   ├── llm_provider.py       # Multi-provider LLM interface (NEW!)
│   ├── logger.py             # Logging configuration (NEW!)
│   ├── summarization.py      # Text summarization (enhanced)
│   ├── transcription.py      # Speech-to-text (improved)
│   ├── translation.py        # Multi-language translation
│   ├── upscaling.py          # Video upscaling (secure)
│   ├── validators.py         # Input validation (NEW!)
│   ├── video.py              # Video processing
│   └── youtube.py            # YouTube download
│
├── models/                    # AI models (download here)
│   └── vosk-model-*/         # Vosk speech recognition
│
├── output_directory/          # Generated outputs
│   ├── *.txt                 # Text summaries
│   ├── *.wav                 # Audio files
│   └── *.log                 # Application logs (NEW!)
│
└── streamlit/                 # Temporary working files
    ├── audio.mp3
    ├── formatted_audio.wav
    ├── upscaled_video.mp4
    └── upscaled_video_h264.mp4
```

## 🛠️ Installation

### Prerequisites

- **Python**: 3.8 or higher
- **FFmpeg**: Required for audio/video processing
- **CUDA**: Optional but recommended for GPU acceleration
- **Disk Space**: ~10GB for models (varies by provider)

### Step 1: Clone the Repository

```bash
git clone https://github.com/koopatroopa787/AMD-pervasive-AI.git
cd AMD-pervasive-AI
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Step 5: Download Vosk Model

```bash
# Create models directory
mkdir -p models

# Download and extract English model
cd models
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip
unzip vosk-model-en-us-0.22-lgraph.zip
cd ..
```

### Step 6: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your preferred editor
nano .env  # or vim, code, etc.
```

## ⚙️ Configuration

### Choose Your AI Provider

Edit `.env` and configure your preferred AI provider:

#### Option 1: Hugging Face (Local, Free, No API Key)
```env
LLM_PROVIDER=huggingface
HF_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2
HF_USE_QUANTIZATION=true
HF_LOAD_IN_4BIT=true
```

#### Option 2: OpenAI
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4
```

#### Option 3: Anthropic Claude
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-api-key-here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

#### Option 4: Ollama (Local, Free)
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434

# Make sure Ollama is running:
# ollama serve
# ollama pull llama3.1
```

#### Option 5: Groq (Fast Cloud API)
```env
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL=mixtral-8x7b-32768
```

### Key Configuration Options

See `.env.example` for all available options. Important settings:

```env
# Paths
VOSK_MODEL_PATH=./models/vosk-model-en-us-0.22-lgraph
OUTPUT_DIRECTORY=./output_directory
TEMP_DIRECTORY=./streamlit

# Performance
USE_GPU=true
ENABLE_CACHING=true
UPSCALE_MAX_WORKERS=4

# Logging
ENABLE_LOGGING=true
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# UI Customization
PAGE_TITLE=AI Video Processor
PAGE_ICON=🎬
```

## 🚀 Usage

### Start the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using Transcript Summarizer

1. **Select Mode**: Choose "📝 Transcript Summarizer" in sidebar
2. **Choose AI Provider**: Select from available providers (top of sidebar)
3. **Select Language**: Pick target language for translation
4. **Input Video**: YouTube URL or upload file
5. **Process**: Click process button and wait
6. **View Results**: Summaries appear in tabs
7. **Download**: Save text and audio files

### Using Video Upscaler

1. **Select Mode**: Choose "🎬 Video Upscaler" in sidebar
2. **Input Video**: YouTube URL or upload file
3. **Process**: Click upscale (may take several minutes)
4. **Preview**: Watch upscaled video
5. **Download**: Save enhanced video

## 🎯 Features

### Video Transcript Summarizer
- ✅ YouTube URL and file upload support
- ✅ Offline speech recognition (Vosk)
- ✅ AI summarization with 5+ provider options
- ✅ Multi-language translation (10+ languages)
- ✅ Audio generation for translations
- ✅ Full, short, and key points summaries
- ✅ Export all outputs (text + audio)

### AI Video Upscaler
- ✅ Real-ESRGAN AI upscaling (2x, 4x, 8x)
- ✅ Parallel frame processing
- ✅ H.264 conversion
- ✅ Configurable quality settings
- ✅ GPU acceleration support

## 🔧 Advanced Configuration

### Using Different Models

```env
# Larger Hugging Face model (requires more RAM)
HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct

# Smaller, faster model
HF_MODEL_ID=microsoft/phi-3-mini-4k-instruct

# Custom Ollama model
OLLAMA_MODEL=mistral:latest
```

### Performance Tuning

```env
# More workers for faster upscaling (uses more CPU/GPU)
UPSCALE_MAX_WORKERS=8

# Longer video processing (default 15s)
UPSCALE_MAX_DURATION=30

# Larger text chunks (faster but less context)
LLM_CHUNK_SIZE=1500
```

### Debugging

```env
# Enable detailed logging
LOG_LEVEL=DEBUG

# Check logs in output_directory/app_YYYYMMDD.log
```

## 🐛 Troubleshooting

### "Configuration validation failed"
- Check your `.env` file exists
- Verify API keys are correct
- Ensure no extra spaces around values

### "FFmpeg not found"
```bash
# Verify installation
ffmpeg -version

# Install if missing (see Installation section)
```

### "Vosk model not found"
- Check `VOSK_MODEL_PATH` in `.env`
- Verify model directory exists: `ls models/`
- Re-download if corrupted (see Installation)

### "CUDA out of memory"
```env
# Enable 4-bit quantization
HF_USE_QUANTIZATION=true
HF_LOAD_IN_4BIT=true

# Or use smaller model
HF_MODEL_ID=microsoft/phi-3-mini-4k-instruct
```

### "Provider not available"
- Check API key is set in `.env`
- For Ollama: ensure `ollama serve` is running
- Check logs: `tail -f output_directory/app_*.log`

## 📊 Performance Notes

**Transcript Summarization** (10-minute video):
- Transcription: ~5-10 minutes
- Summarization: 1-5 minutes (varies by provider)
- Translation: ~30 seconds

**Video Upscaling** (15 seconds, 4x):
- With GPU: ~2-5 minutes
- CPU only: ~10-20 minutes

*Performance varies based on hardware and selected models*

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - Web framework
- [Vosk](https://alphacephei.com/vosk/) - Speech recognition
- [Real-ESRGAN](https://github.com/ai-forever/Real-ESRGAN) - Video upscaling
- [Hugging Face](https://huggingface.co/) - Model hub
- [OpenAI](https://openai.com/), [Anthropic](https://anthropic.com/), [Ollama](https://ollama.ai/), [Groq](https://groq.com/) - AI providers

## 🔮 Future Roadmap

- [ ] More language support
- [ ] Batch video processing
- [ ] Subtitle generation (SRT format)
- [ ] Custom model fine-tuning
- [ ] API endpoint for programmatic access
- [ ] Docker containerization
- [ ] Video chapter detection
- [ ] Multiple video quality presets

---

**Built with ❤️ by the AMD Pervasive AI Team**
