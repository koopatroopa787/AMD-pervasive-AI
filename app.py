"""
AI Video Processor - Enhanced Streamlit Application
Supports video summarization, translation, and upscaling with multiple AI providers.
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import streamlit as st
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration and utilities
from config import get_config, AppConfig
from utils.logger import setup_logging, get_logger
from utils.llm_provider import get_llm_provider, LLMProviderFactory
from utils.youtube import get_video_info, download_youtube_video
from utils.audio import extract_audio, convert_audio_format
from utils.transcription import transcribe_audio_vosk
from utils.summarization import generate_summary
from utils.upscaling import upscale_video, convert_to_h264
from utils.translation import translate_text_to_text, translate_text_to_audio, get_supported_languages
from utils.validators import (
    is_valid_youtube_url, validate_video_file, sanitize_filename
)

# Initialize configuration and logging
try:
    config = get_config()

    if config.enable_logging:
        log_file = config.paths.output_directory / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        setup_logging(
            log_level=config.log_level,
            log_file=log_file,
            enable_console=True
        )

    logger = get_logger(__name__)
    logger.info("=" * 80)
    logger.info("AI Video Processor Starting")
    logger.info(f"Configuration: {config.llm.provider} provider")
    logger.info("=" * 80)

except Exception as e:
    st.error(f"Configuration error: {e}")
    st.stop()


# Page configuration
st.set_page_config(
    page_title=config.page_title,
    page_icon=config.page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# Application header
st.markdown(f'<div class="main-header">{config.page_icon} {config.page_title}</div>', unsafe_allow_html=True)
st.markdown("---")


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # LLM Provider Selection
    st.subheader("🤖 AI Model Provider")

    available_providers = LLMProviderFactory.get_available_providers(config)

    if not available_providers:
        st.error("No AI providers are configured! Please check your .env file.")
        st.stop()

    current_provider = st.selectbox(
        "Select LLM Provider",
        available_providers,
        index=available_providers.index(config.llm.provider) if config.llm.provider in available_providers else 0,
        help="Choose which AI model to use for summarization"
    )

    # Update config if provider changed
    if current_provider != config.llm.provider:
        config.llm.provider = current_provider
        if 'llm_provider' in st.session_state:
            del st.session_state['llm_provider']

    # Display current model info
    st.info(f"**Active Model:** {config._get_active_model_name()}")

    st.markdown("---")

    # Language selection
    st.subheader("🌍 Translation Language")
    supported_languages = get_supported_languages()
    selected_language = st.selectbox(
        "Select output language",
        supported_languages,
        help="Language for translated summaries and audio"
    )

    st.markdown("---")

    # Process type selection
    st.subheader("🎯 Processing Mode")
    process_type = st.radio(
        "Select what you want to do",
        ("📝 Transcript Summarizer", "🎬 Video Upscaler"),
        help="Choose between AI summarization or video upscaling"
    )

    st.markdown("---")

    # System info
    with st.expander("ℹ️ System Information"):
        st.write(f"**GPU Available:** {'Yes' if config.use_gpu else 'No'}")
        st.write(f"**Caching:** {'Enabled' if config.enable_caching else 'Disabled'}")
        st.write(f"**Log Level:** {config.log_level}")
        st.write(f"**Output Dir:** {config.paths.output_directory}")

    # Configuration display
    with st.expander("🔧 Advanced Settings"):
        st.json(config.to_dict())


def process_transcript_summarizer():
    """Handle transcript summarization workflow"""
    st.header("📝 Video Transcript Summarizer")
    st.markdown("Extract audio, transcribe speech, and generate AI-powered summaries in multiple languages.")

    # Input method selection
    col1, col2 = st.columns(2)

    with col1:
        input_method = st.radio(
            "Select input method",
            ("🔗 YouTube URL", "📁 Upload Video File"),
            horizontal=True
        )

    video_path = None
    video_info = {}

    # YouTube URL input
    if input_method == "🔗 YouTube URL":
        url = st.text_input(
            "Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a valid YouTube video URL"
        )

        if st.button("🚀 Download and Process", type="primary", use_container_width=True):
            if not url:
                st.error("Please enter a YouTube URL")
                return

            if not is_valid_youtube_url(url):
                st.error("Invalid YouTube URL format")
                return

            try:
                with st.spinner("📥 Fetching video information..."):
                    video_info = get_video_info(url)
                    logger.info(f"Retrieved video info: {video_info.get('title', 'Unknown')}")

                # Display video metadata
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(video_info["thumbnail_url"], use_container_width=True)

                with col2:
                    st.markdown(f"**📺 Title:** {video_info['title']}")
                    st.markdown(f"**👤 Author:** {video_info['author']}")
                    st.markdown(f"**📅 Published:** {video_info['publish_date']}")
                    st.markdown(f"**👁️ Views:** {video_info['views']:,}")
                    with st.expander("📄 Description"):
                        st.write(video_info['description'])

                st.markdown('</div>', unsafe_allow_html=True)

                # Download video
                with st.spinner("⬇️ Downloading video..."):
                    video_path = download_youtube_video(url, str(config.paths.temp_directory))
                    logger.info(f"Video downloaded: {video_path}")

                st.success("✅ Video downloaded successfully!")

            except Exception as e:
                logger.error(f"YouTube download failed: {e}")
                st.error(f"❌ Failed to download video: {e}")
                return

    # File upload input
    elif input_method == "📁 Upload Video File":
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            help="Maximum file size: 500MB"
        )

        if uploaded_file is not None:
            try:
                # Save uploaded file
                video_filename = sanitize_filename(uploaded_file.name)
                video_path = config.paths.temp_directory / video_filename

                with st.spinner("💾 Saving uploaded file..."):
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                # Validate file
                is_valid, error_msg = validate_video_file(video_path)
                if not is_valid:
                    st.error(f"❌ Invalid video file: {error_msg}")
                    return

                st.success(f"✅ File uploaded: {video_filename}")
                logger.info(f"Video uploaded: {video_path}")

            except Exception as e:
                logger.error(f"File upload failed: {e}")
                st.error(f"❌ Failed to save file: {e}")
                return

    # Process video if available
    if video_path:
        st.markdown("---")
        st.subheader("🔄 Processing Pipeline")

        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Extract audio
            status_text.text("🎵 Extracting audio from video...")
            progress_bar.progress(10)

            audio_path = config.paths.temp_directory / "audio.mp3"
            extract_audio(video_path, audio_path)
            logger.info("Audio extracted successfully")

            # Step 2: Convert audio format
            status_text.text("🔊 Converting audio format...")
            progress_bar.progress(20)

            formatted_audio_path = config.paths.temp_directory / "formatted_audio.wav"
            convert_audio_format(audio_path, formatted_audio_path)
            logger.info("Audio format converted")

            # Step 3: Transcribe audio
            status_text.text("🎤 Transcribing audio (this may take a while)...")
            progress_bar.progress(30)

            transcript = transcribe_audio_vosk(formatted_audio_path)

            if not transcript:
                st.error("❌ Transcription failed - no text was extracted")
                return

            progress_bar.progress(50)
            logger.info(f"Transcription complete: {len(transcript)} characters")

            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.success(f"✅ Transcription complete! ({len(transcript.split())} words)")
            st.markdown('</div>', unsafe_allow_html=True)

            # Display transcript
            with st.expander("📄 View Full Transcript"):
                st.text_area("Transcript", transcript, height=200)

            # Step 4: Generate summaries
            status_text.text("🤖 Generating AI summaries...")
            progress_bar.progress(60)

            # Get LLM provider
            if 'llm_provider' not in st.session_state:
                st.session_state.llm_provider = get_llm_provider(config)

            llm_provider = st.session_state.llm_provider

            summaries = generate_summary(transcript, video_info, llm_provider)

            progress_bar.progress(75)
            logger.info("Summaries generated successfully")

            # Step 5: Translate summaries
            status_text.text(f"🌍 Translating to {selected_language}...")
            progress_bar.progress(85)

            translated_summaries = {
                'full': translate_text_to_text(summaries['full'], selected_language),
                'short': translate_text_to_text(summaries['short'], selected_language),
                'key_points': translate_text_to_text(summaries['key_points'], selected_language)
            }

            # Step 6: Generate audio
            status_text.text("🔊 Generating translated audio...")
            progress_bar.progress(95)

            translated_audio = {
                'full': translate_text_to_audio(translated_summaries['full'], selected_language),
                'short': translate_text_to_audio(translated_summaries['short'], selected_language),
                'key_points': translate_text_to_audio(translated_summaries['key_points'], selected_language)
            }

            # Save outputs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = config.paths.output_directory

            # Save text files
            for key, content in {**summaries, **{'translated_' + k: v for k, v in translated_summaries.items()}}.items():
                file_path = output_dir / f"{key}_{timestamp}.txt"
                file_path.write_text(content, encoding='utf-8')

            # Save audio files
            audio_files = {}
            for key, audio_data in translated_audio.items():
                file_path = output_dir / f"translated_{key}_{timestamp}.wav"
                sf.write(file_path, audio_data, 16000)
                audio_files[key] = file_path

            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")

            logger.info("All processing complete")

            st.markdown("---")
            st.header("📊 Results")

            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📝 Full Summary", "⚡ Short Summary", "🎯 Key Points", "🌍 Translations"])

            with tab1:
                st.subheader("Full Summary")
                st.write(summaries['full'])
                st.download_button("⬇️ Download", summaries['full'], f"full_summary_{timestamp}.txt")

            with tab2:
                st.subheader("Short Summary")
                st.info(summaries['short'])
                st.download_button("⬇️ Download", summaries['short'], f"short_summary_{timestamp}.txt")

            with tab3:
                st.subheader("Key Points")
                st.write(summaries['key_points'])
                st.download_button("⬇️ Download", summaries['key_points'], f"key_points_{timestamp}.txt")

            with tab4:
                st.subheader(f"Translations ({selected_language})")

                st.markdown("**Full Summary:**")
                st.write(translated_summaries['full'])
                st.audio(str(audio_files['full']))

                st.markdown("**Short Summary:**")
                st.write(translated_summaries['short'])
                st.audio(str(audio_files['short']))

                st.markdown("**Key Points:**")
                st.write(translated_summaries['key_points'])
                st.audio(str(audio_files['key_points']))

            st.balloons()

        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            st.error(f"❌ Processing failed: {e}")


def process_video_upscaler():
    """Handle video upscaling workflow"""
    st.header("🎬 AI Video Upscaler")
    st.markdown("Enhance video quality using AI upscaling (Real-ESRGAN).")

    # Input method selection
    input_method = st.radio(
        "Select input method",
        ("🔗 YouTube URL", "📁 Upload Video File"),
        horizontal=True
    )

    video_path = None

    # YouTube URL input
    if input_method == "🔗 YouTube URL":
        url = st.text_input(
            "Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a valid YouTube video URL"
        )

        if st.button("🚀 Download and Upscale", type="primary", use_container_width=True):
            if not url or not is_valid_youtube_url(url):
                st.error("Please enter a valid YouTube URL")
                return

            try:
                with st.spinner("⬇️ Downloading video..."):
                    video_path = download_youtube_video(url, str(config.paths.temp_directory))
                    st.success("✅ Video downloaded!")

            except Exception as e:
                logger.error(f"Download failed: {e}")
                st.error(f"❌ Download failed: {e}")
                return

    # File upload input
    elif input_method == "📁 Upload Video File":
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "mov", "avi", "mkv"],
            help="Note: Large files may take a long time to process"
        )

        if uploaded_file is not None:
            try:
                video_filename = sanitize_filename(uploaded_file.name)
                video_path = config.paths.temp_directory / video_filename

                with st.spinner("💾 Saving file..."):
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                st.success(f"✅ File uploaded: {video_filename}")

            except Exception as e:
                st.error(f"❌ Upload failed: {e}")
                return

    # Process video if available
    if video_path:
        st.markdown("---")

        try:
            with st.spinner("🔄 Loading upscaling model..."):
                from huggingface_hub import hf_hub_download
                from RealESRGAN import RealESRGAN
                import torch

                model_path = hf_hub_download(
                    repo_id=config.upscaling.model_id,
                    filename=config.upscaling.model_filename
                )

                device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
                model = RealESRGAN(device, scale=config.upscaling.scale)
                model.load_weights(model_path, download=True)

                logger.info(f"Model loaded on {device}")

            # Upscale video
            output_video_path = config.paths.temp_directory / 'upscaled_video.mp4'
            converted_video_path = config.paths.temp_directory / 'upscaled_video_h264.mp4'

            with st.spinner(f"⬆️ Upscaling video ({config.upscaling.scale}x)..."):
                upscale_video(
                    video_path,
                    output_video_path,
                    model,
                    max_duration=config.upscaling.max_duration,
                    max_workers=config.upscaling.max_workers
                )

            with st.spinner("🎞️ Converting to H.264..."):
                convert_to_h264(output_video_path, converted_video_path)

            st.success("✅ Video upscaled successfully!")

            # Display result
            st.subheader("📺 Upscaled Video")
            st.video(str(converted_video_path))

            # Download button
            with open(converted_video_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Upscaled Video",
                    f,
                    file_name=f"upscaled_{Path(video_path).stem}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

            st.balloons()

        except Exception as e:
            logger.error(f"Upscaling failed: {e}", exc_info=True)
            st.error(f"❌ Upscaling failed: {e}")


# Main application logic
def main():
    """Main application entry point"""
    try:
        if "📝 Transcript" in process_type:
            process_transcript_summarizer()
        else:
            process_video_upscaler()

    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
