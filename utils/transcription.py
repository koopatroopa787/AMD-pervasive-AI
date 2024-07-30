import wave
import json
from vosk import Model, KaldiRecognizer

def transcribe_audio_vosk(audio_path):
    model_path = "/home/ubuntu/streamlit/vosk-model-en-us-0.22-lgraph"  # Update this to your actual model path
    model = Model(model_path)
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    
    transcript = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            transcript += json.loads(result)['text'] + " "
    
    return transcript
