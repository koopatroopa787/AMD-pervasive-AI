import torchaudio
from transformers import AutoProcessor, SeamlessM4TModel

processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

def translate_text_to_text(text, target_language):
    text_inputs = processor(text, return_tensors="pt", padding=True)
    output_tokens = model.generate(**text_inputs, tgt_lang=target_language, generate_speech=False)
    translated_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    return translated_text

def translate_text_to_audio(text, target_language):
    text_inputs = processor(text, return_tensors="pt", padding=True)
    audio_array = model.generate(**text_inputs, tgt_lang=target_language)[0].cpu().numpy().squeeze()
    return audio_array

def get_supported_languages():
    return [
        "eng", "fra", "deu", "spa", "ita", "por", "rus", "zho", "jpn", "kor"
        # Add other supported languages as needed
    ]
