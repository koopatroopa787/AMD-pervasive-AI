import os
from pathlib import Path
import streamlit as st
from utils.youtube import get_video_info, download_youtube_video
from utils.audio import extract_audio, convert_audio_format
from utils.transcription import transcribe_audio_vosk
from utils.summarization import chunk_text, summarize_text_mistral, query_model
from utils.upscaling import upscale_video, convert_to_h264
from utils.video import extract_frames, save_video
from utils.translation import translate_text_to_text, translate_text_to_audio, get_supported_languages
import torchaudio
import numpy as np
import soundfile as sf

# Streamlit App
st.title("Video Processor")

# Language selection
supported_languages = get_supported_languages()
selected_language = st.sidebar.selectbox("Select language for translation", supported_languages)

# Input for YouTube URL or file upload
st.sidebar.title("Choose an Option")
option = st.sidebar.radio("Select process type", ("Transcript Summarizer", "Video Upscaler"))

transcript = ""
video_info = {}

if option == "Transcript Summarizer":
    input_type = st.radio("Select input type", ("YouTube URL", "Upload Video"))
    if input_type == "YouTube URL":
        url = st.text_input("Enter YouTube URL")
        if st.button("Download and Process"):
            video_info = get_video_info(url)
            st.image(video_info["thumbnail_url"], caption="Video Thumbnail")
            st.write(f"**Title:** {video_info['title']}")
            st.write(f"**Author:** {video_info['author']}")
            st.write(f"**Description:** {video_info['description']}")
            st.write(f"**Publish Date:** {video_info['publish_date']}")
            st.write(f"**Views:** {video_info['views']}")

            # Ensure output directory exists
            output_directory = "./streamlit"
            os.makedirs(output_directory, exist_ok=True)
            
            video_path = download_youtube_video(url, output_directory)
            audio_path = os.path.join(output_directory, "audio.mp3")
            formatted_audio_path = os.path.join(output_directory, "formatted_audio.wav")
            extract_audio(video_path, audio_path)
            
            # Check if audio file was created
            if os.path.exists(audio_path):
                convert_audio_format(audio_path, formatted_audio_path)
                transcript = transcribe_audio_vosk(formatted_audio_path)
            else:
                st.error("Audio extraction failed. Please check the video and try again.")

    elif input_type == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            # Ensure output directory exists
            output_directory = "./streamlit"
            os.makedirs(output_directory, exist_ok=True)
            
            video_path = os.path.join(output_directory, "uploaded_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            audio_path = os.path.join(output_directory, "audio.mp3")
            formatted_audio_path = os.path.join(output_directory, "formatted_audio.wav")

            extract_audio(video_path, audio_path)
            
            # Check if audio file was created
            if os.path.exists(audio_path):
                convert_audio_format(audio_path, formatted_audio_path)
                transcript = transcribe_audio_vosk(formatted_audio_path)
            else:
                st.error("Audio extraction failed. Please check the video and try again.")

    # Process transcript if available
    if transcript:
        st.write("Transcript generated successfully. Processing for summary...")

        # Chunk text
        chunks = chunk_text(transcript)

        # Load the model only after transcript is created and chunked
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Configuration for 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=quantization_config,
            device_map="auto"
        )

        # Summarize each chunk
        summaries = [summarize_text_mistral(chunk, video_info, model, tokenizer) for chunk in chunks]
        final_summary = " ".join(summaries).replace('\n', ' ').replace('  ', ' ')

        # Generate a short summary and key points
        short_summary = query_model("What is this video about? Provide a detailed and coherent summary.", final_summary, model, tokenizer)
        key_points = query_model("List 7-9 key points from this video.", final_summary, model, tokenizer)

        # Save summaries
        output_directory = Path("/home/ubuntu/streamlit/output_directory")
        os.makedirs(output_directory, exist_ok=True)
        
        final_summary_output_path = output_directory / "final_summary.txt"
        if os.path.exists(final_summary_output_path):
            os.remove(final_summary_output_path)
        with open(final_summary_output_path, "w") as file:
            file.write(final_summary)

        short_summary_path = output_directory / "short_summary.txt"
        if os.path.exists(short_summary_path):
            os.remove(short_summary_path)
        with open(short_summary_path, "w") as file:
            file.write(short_summary)

        key_points_path = output_directory / "key_points.txt"
        if os.path.exists(key_points_path):
            os.remove(key_points_path)
        with open(key_points_path, "w") as file:
            file.write(key_points)

        # Display summaries
        st.subheader("Final Summary")
        st.write(final_summary)

        st.subheader("Short Summary")
        st.write(short_summary)

        st.subheader("Key Points")
        st.write(key_points)

        # Translate summaries
        translated_summary = translate_text_to_text(final_summary, selected_language)
        translated_short_summary = translate_text_to_text(short_summary, selected_language)
        translated_key_points = translate_text_to_text(key_points, selected_language)

        st.subheader("Translated Final Summary")
        st.write(translated_summary)

        st.subheader("Translated Short Summary")
        st.write(translated_short_summary)

        st.subheader("Translated Key Points")
        st.write(translated_key_points)

        # Generate and play audio
        translated_summary_audio = translate_text_to_audio(translated_summary, selected_language)
        translated_short_summary_audio = translate_text_to_audio(translated_short_summary, selected_language)
        translated_key_points_audio = translate_text_to_audio(translated_key_points, selected_language)

        # Save and play audio files
        summary_audio_path = output_directory / "translated_summary.wav"
        short_summary_audio_path = output_directory / "translated_short_summary.wav"
        key_points_audio_path = output_directory / "translated_key_points.wav"

        sf.write(summary_audio_path, translated_summary_audio, 16000)
        sf.write(short_summary_audio_path, translated_short_summary_audio, 16000)
        sf.write(key_points_audio_path, translated_key_points_audio, 16000)

        st.audio(str(summary_audio_path))
        st.audio(str(short_summary_audio_path))
        st.audio(str(key_points_audio_path))

elif option == "Video Upscaler":
    input_type = st.radio("Select input type", ("YouTube URL", "Upload Video"))
    if input_type == "YouTube URL":
        url = st.text_input("Enter YouTube URL")
        if st.button("Download and Upscale"):
            video_path = download_youtube_video(url, "./")
            output_video_path = './streamlit/upscaled_video.mp4'
            converted_video_path = './streamlit/upscaled_video_h264.mp4'
            
            from huggingface_hub import hf_hub_download
            from RealESRGAN import RealESRGAN
            import torch

            model_id = "ai-forever/Real-ESRGAN"
            model_filename = "RealESRGAN_x4.pth"
            model_path = hf_hub_download(repo_id=model_id, filename=model_filename)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = RealESRGAN(device, scale=4)
            model.load_weights(model_path, download=True)

            upscale_video(video_path, output_video_path, model)
            convert_to_h264(output_video_path, converted_video_path)
            st.video(converted_video_path)
    elif input_type == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            video_path = "./streamlit/video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            output_video_path = 'upscaled_video.mp4'
            converted_video_path = 'upscaled_video_h264.mp4'
            
            model_id = "ai-forever/Real-ESRGAN"
            model_filename = "RealESRGAN_x4.pth"
            model_path = hf_hub_download(repo_id=model_id, filename=model_filename)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = RealESRGAN(device, scale=4)
            model.load_weights(model_path, download=True)

            upscale_video(video_path, output_video_path, model)
            convert_to_h264(output_video_path, converted_video_path)
            st.video(converted_video_path)
